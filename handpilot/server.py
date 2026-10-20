"""The window: a small local web server, and the page it serves.

A web page rather than a native toolkit, because it is the one UI layer every
Windows machine already has a modern renderer for, it needs nothing installed,
and it opens in an Edge (or Chrome) app window that looks like any other app.
The control loop never touches it: the page reads snapshots 30 times a second
and the link thread does not wait for anyone.

Security, since this is a local server that can move your mouse:

* it binds to 127.0.0.1 only;
* every API call must carry a token generated fresh for each run, which only
  the window it opens is given -- other web pages in your browser cannot read
  it, so they cannot drive the app;
* the Host header must be 127.0.0.1 or localhost, which defeats DNS rebinding;
* the one unauthenticated data endpoint, ``/stream``, is off unless you turn on
  "Browser stream", and it only ever sends landmarks.
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
import webbrowser
from urllib.parse import parse_qs, urlsplit

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
LIVE_HZ = 30
STREAM_HZ = 120
#: How long the app waits for a closed window to come back (a page reload)
#: before it takes the close as "quit".
CLOSE_GRACE_S = 4.0


class UIServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second process bind the same port, which
    # would defeat the single-instance check in __main__.
    allow_reuse_address = False

    def __init__(self, app, port: int, quit_on_close: bool = True) -> None:
        self.app = app
        self.token = secrets.token_urlsafe(18)
        self.quit_on_close = quit_on_close
        self._clients = 0
        self._ever_connected = False
        self._clients_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    def client_joined(self) -> None:
        with self._clients_lock:
            self._clients += 1
            self._ever_connected = True

    def client_left(self) -> None:
        with self._clients_lock:
            self._clients -= 1
            last = self._clients == 0
        if last and self.quit_on_close:
            timer = threading.Timer(CLOSE_GRACE_S, self._quit_if_still_closed)
            timer.daemon = True
            timer.start()

    def _quit_if_still_closed(self) -> None:
        with self._clients_lock:
            closed = self._ever_connected and self._clients == 0
        if closed:
            self.app.controller.log("window closed, shutting down")
            self.app.quit_event.set()

    def open_window(self) -> str:
        return open_window(self.url)


class Handler(http.server.BaseHTTPRequestHandler):
    server: UIServer
    server_version = "HandPilot"

    def log_message(self, *args) -> None:       # quiet: the console is for the user
        pass

    # ----------------------------------------------------------------- helpers

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.port}", f"localhost:{self.server.port}")

    def _token_ok(self, query: dict) -> bool:
        given = self.headers.get("X-HandPilot-Token") or (query.get("t") or [""])[0]
        return secrets.compare_digest(given, self.server.token)

    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _start_events(self, cors: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    # ---------------------------------------------------------------- routing

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if not self._host_ok():
            return self._json(403, {"error": "bad host"})

        if url.path in STATIC:
            name, kind = STATIC[url.path]
            try:
                with open(os.path.join(WEB_DIR, name), "rb") as handle:
                    body = handle.read()
            except OSError:
                return self._json(404, {"error": "missing file"})
            return self._send(200, body, kind,
                              {"Content-Security-Policy": "default-src 'self'; "
                                                          "img-src 'self' data:; "
                                                          "style-src 'self' 'unsafe-inline'"})
        if url.path == "/api/ping":
            return self._json(200, {"app": "handpilot"})
        if url.path == "/api/focus":
            self.server.open_window()
            return self._json(200, {"app": "handpilot"})
        if url.path == "/stream":
            return self._public_stream()

        if not self._token_ok(query):
            return self._json(401, {"error": "missing or wrong token"})
        if url.path == "/api/state":
            state = self.server.app.state()
            state["stream_url"] = f"http://127.0.0.1:{self.server.port}/stream"
            return self._json(200, state)
        if url.path == "/api/live":
            return self._live()
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        if not self._host_ok():
            return self._json(403, {"error": "bad host"})
        if not self._token_ok({}):
            return self._json(401, {"error": "missing or wrong token"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65536:
                return self._json(413, {"error": "too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad json"})
        if not isinstance(body, dict):
            return self._json(400, {"error": "expected an object"})

        app = self.server.app
        if url.path == "/api/config":
            ok, error = app.set_setting(str(body.get("path", "")), body.get("value"))
            # The whole config comes back, so the window shows what was actually
            # stored -- including anything clamped or adjusted alongside.
            return self._json(200 if ok else 400, {"ok": ok, "error": error, "config": app.cfg})
        if url.path == "/api/command":
            result = app.command(str(body.get("cmd", "")), body.get("value"))
            return self._json(200 if result.get("ok") else 400, result)
        return self._json(404, {"error": "not found"})

    # ----------------------------------------------------------------- streams

    def _live(self) -> None:
        """Server-sent events: one snapshot per tick for the window."""
        app = self.server.app
        self._start_events()
        self.server.client_joined()
        last_rev = -1
        try:
            while not app.quit_event.is_set():
                snap = app.live()
                if snap["log_rev"] != last_rev:
                    last_rev = snap["log_rev"]
                    snap["log"] = list(app.controller.log_lines)[-80:]
                self.wfile.write(b"data: " + json.dumps(snap).encode("utf-8") + b"\n\n")
                self.wfile.flush()
                time.sleep(1.0 / LIVE_HZ)
        except (ConnectionError, OSError):
            pass
        finally:
            self.server.client_left()

    def _public_stream(self) -> None:
        """Opt-in raw landmarks for other pages and apps: ``new EventSource(
        "http://127.0.0.1:9870/stream")``. Sends each new frame once."""
        app = self.server.app
        if not app.cfg.get("stream", {}).get("browser"):
            return self._json(403, {"error": "browser stream is off; turn it on in HandPilot"})
        self._start_events(cors=True)
        last_seq = None
        try:
            while not app.quit_event.is_set() and app.cfg["stream"].get("browser"):
                frame, _ = app.controller.snapshot()
                if frame is not None and frame.seq != last_seq:
                    last_seq = frame.seq
                    payload = json.dumps(frame.to_dict(), separators=(",", ":"))
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                time.sleep(1.0 / STREAM_HZ)
        except (ConnectionError, OSError):
            pass


# ------------------------------------------------------------------- windows

def _app_browsers() -> list[str]:
    """Chromium browsers that can open a page as a chrome-less app window."""
    found = []
    env = os.environ.get
    for root in (env("ProgramFiles(x86)"), env("ProgramFiles"), env("LOCALAPPDATA")):
        if not root:
            continue
        for rel in (r"Microsoft\Edge\Application\msedge.exe",
                    r"Google\Chrome\Application\chrome.exe",
                    r"BraveSoftware\Brave-Browser\Application\brave.exe"):
            path = os.path.join(root, rel)
            if os.path.exists(path):
                found.append(path)
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        path = shutil.which(name)
        if path:
            found.append(path)
    return found


def open_window(url: str) -> str:
    """Open the UI as an app window if a Chromium browser is around, otherwise
    as a normal browser tab. Returns what it used."""
    profile = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                           "HandPilot", "window")
    for exe in _app_browsers():
        try:
            subprocess.Popen(
                [exe, f"--app={url}", f"--user-data-dir={profile}",
                 "--window-size=1320,880", "--no-first-run", "--no-default-browser-check"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return os.path.basename(exe)
        except OSError:
            continue
    webbrowser.open(url)
    return "browser"


def ping_existing(port: int) -> bool:
    """True if another HandPilot already owns ``port`` -- in which case it has
    just been asked to open a window, and this process can step aside."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/focus", timeout=1.5) as reply:
            return json.load(reply).get("app") == "handpilot"
    except (OSError, ValueError):
        return False

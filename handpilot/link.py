"""The connection to the phone: one thread that dials, reads, and reconnects.

Frames are handed to the consumer from *this* thread rather than posted to a
queue for someone else to pick up. A queue would be tidier and would cost a
scheduler wakeup per frame -- roughly a millisecond of jitter on Windows, on a
budget where the entire phone-side pipeline is under ten.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from . import protocol, usbmux


@dataclass
class LinkStats:
    connected: bool = False
    transport: str = "usb"
    device: str = ""
    error: str = ""
    frames: int = 0
    fps: float = 0.0
    #: Round trip through the cable, measured with an echo so it needs no
    #: synchronised clock between the two machines.
    rtt_ms: float = 0.0
    #: What the phone reported about itself, most recent status message.
    phone: dict = field(default_factory=dict)


class Link:
    def __init__(self, on_frame, on_status=None) -> None:
        self._on_frame = on_frame
        self._on_status = on_status
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.stats = LinkStats()

        self._transport = "usb"
        self._host = ""
        self._port = 9871
        self._pending_ping: tuple[int, float] | None = None
        self._ping_seq = 0

    # ------------------------------------------------------------------ control

    def start(self, transport: str, host: str, port: int) -> None:
        self.stop()
        self._transport, self._host, self._port = transport, host, int(port)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="handpilot-link", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self.stats.connected = False

    def send_command(self, obj: dict) -> None:
        with self._lock:
            sock = self._sock
        if sock is None:
            return
        try:
            sock.sendall(protocol.encode_command(obj))
        except OSError:
            pass

    def ping(self) -> None:
        self._ping_seq += 1
        self._pending_ping = (self._ping_seq, time.perf_counter())
        self.send_command({"ping": self._ping_seq})

    # --------------------------------------------------------------------- loop

    def _run(self) -> None:
        backoff = 0.4
        while not self._stop.is_set():
            try:
                sock = self._dial()
            except Exception as exc:                      # noqa: BLE001 - surfaced in the UI
                self.stats.error = str(exc)
                self.stats.connected = False
                # Retry rather than give up: the usual reason for failing here is
                # that the phone app has not been started yet, and waiting is a
                # better answer than making the user press Connect again.
                if self._stop.wait(min(backoff, 3.0)):
                    break
                backoff = min(backoff * 1.6, 3.0)
                continue

            backoff = 0.4
            with self._lock:
                self._sock = sock
            self.stats.connected = True
            self.stats.error = ""
            try:
                self._pump(sock)
            except Exception as exc:                      # noqa: BLE001
                self.stats.error = str(exc)
            finally:
                self.stats.connected = False
                with self._lock:
                    if self._sock is sock:
                        self._sock = None
                try:
                    sock.close()
                except OSError:
                    pass

    def _dial(self) -> socket.socket:
        if self._transport == "usb":
            sock, device = usbmux.connect_first(self._port)
            self.stats.transport = "usb"
            self.stats.device = device["serial"]
        else:
            if not self._host:
                raise RuntimeError("No phone address set for the Wi-Fi transport")
            sock = socket.create_connection((self._host, self._port), timeout=4.0)
            sock.settimeout(None)
            self.stats.transport = "wifi"
            self.stats.device = self._host
        # Small urgent packets are exactly the case Nagle handles badly: it would
        # hold a frame back waiting for a companion that is never coming.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock

    def _pump(self, sock: socket.socket) -> None:
        reader = protocol.Reader()
        last_fps_stamp = time.perf_counter()
        frames_in_window = 0
        last_ping = 0.0

        while not self._stop.is_set():
            data = sock.recv(65536)
            if not data:
                raise ConnectionError("phone closed the connection")
            now = time.perf_counter()

            for mtype, message in reader.feed(data, now):
                if mtype == protocol.TYPE_FRAME:
                    self.stats.frames += 1
                    frames_in_window += 1
                    self._on_frame(message)
                elif mtype == protocol.TYPE_STATUS:
                    self.stats.phone = message
                    if self._on_status is not None:
                        self._on_status(message)
                elif mtype == protocol.TYPE_PONG:
                    pending = self._pending_ping
                    if pending and message.get("ping") == pending[0]:
                        rtt = (time.perf_counter() - pending[1]) * 1000.0
                        # Lightly smoothed: a single sample catches whatever the
                        # OS scheduler was doing at that instant.
                        self.stats.rtt_ms = (
                            rtt if self.stats.rtt_ms == 0 else self.stats.rtt_ms * 0.7 + rtt * 0.3
                        )
                        self._pending_ping = None

            if now - last_fps_stamp >= 0.5:
                self.stats.fps = frames_in_window / (now - last_fps_stamp)
                frames_in_window = 0
                last_fps_stamp = now
            if now - last_ping >= 1.0:
                last_ping = now
                self.ping()

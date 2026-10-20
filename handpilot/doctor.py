"""Check the path from this PC to the phone, one link at a time.

Run when something is not connecting. Each check names the specific fix rather
than a generic failure, because the usual problems -- no Apple USB service, no
cable, phone locked, app not started -- all present identically as "it does not
connect".
"""

from __future__ import annotations

import socket
import sys
import time


def _ok(text: str) -> None:
    print(f"  [ ok ] {text}")


def _bad(text: str, fix: str = "") -> None:
    print(f"  [FAIL] {text}")
    if fix:
        print(f"         -> {fix}")


def _open(port: int, wifi: str):
    if wifi:
        sock = socket.create_connection((wifi, port), timeout=3.0)
        return sock, wifi
    from . import usbmux
    sock, device = usbmux.connect_first(port)
    return sock, device["serial"]


def main(port: int = 9871, wifi: str = "") -> int:
    print("HandPilot doctor\n")
    failures = 0

    print("Python")
    if sys.version_info < (3, 10):
        _bad(f"Python {sys.version.split()[0]} is too old",
             "install Python 3.10 or newer from python.org")
        failures += 1
    else:
        _ok(f"Python {sys.version.split()[0]}")

    if sys.platform == "win32":
        from .server import _app_browsers
        if _app_browsers():
            _ok("Edge or Chrome found for the app window")
        else:
            _bad("no Edge or Chrome found", "the app will open in your default browser instead")

    if not wifi:
        print("\nApple USB service")
        from . import usbmux
        try:
            devices = usbmux.list_devices()
            _ok("usbmuxd is reachable")
        except usbmux.UsbmuxError as exc:
            _bad(str(exc))
            return failures + 1

        print("\nDevices")
        if not devices:
            _bad("no iPhone attached",
                 "plug it in with a data cable, unlock it, and tap Trust")
            return failures + 1
        for device in devices:
            kind = "USB" if device["usb"] else "Wi-Fi sync (no good for this)"
            _ok(f"{device['serial']}  via {kind}")
        if not any(d["usb"] for d in devices):
            _bad("nothing over the cable",
                 "some cables are charge-only; try the one that came with the phone")
            return failures + 1

    print(f"\nHandPilot app on port {port}")
    try:
        sock, who = _open(port, wifi)
    except Exception as exc:                                  # noqa: BLE001
        _bad(str(exc),
             "open HandPilot on the phone, tap Start, and check the port there matches. "
             "Close the HandPilot desktop app first: the phone serves one connection at a time.")
        return failures + 1

    _ok(f"connected to {who}")
    sock.settimeout(3.0)
    start = time.perf_counter()
    try:
        data = sock.recv(65536)
    except socket.timeout:
        _bad("connected, but the phone is sending nothing",
             "tap Start in the app; its status should read 'streaming'")
        sock.close()
        return failures + 1

    elapsed = (time.perf_counter() - start) * 1000
    if data[:4] == b"HPU1":
        _ok(f"receiving data ({len(data)} bytes, first packet after {elapsed:.0f} ms)")
    else:
        _bad(f"unexpected data on the wire: {data[:16]!r}",
             "something else is listening on that port inside the phone")
        failures += 1

    print("\nThroughput")
    count = 0
    deadline = time.perf_counter() + 1.0
    try:
        while time.perf_counter() < deadline:
            chunk = sock.recv(65536)
            if not chunk:
                break
            count += chunk.count(b"HPU1")
    except socket.timeout:
        pass
    sock.close()
    if count >= 50:
        _ok(f"{count} messages/second")
    elif count > 0:
        _bad(f"only {count} messages/second",
             "expect 60-120; check the frame rate, and whether the room is bright "
             "enough for the shutter speed you chose")
        failures += 1
    else:
        _bad("no messages", "tap Start in the phone app")
        failures += 1

    print("\nAll good." if not failures else f"\n{failures} problem(s) found.")
    return 1 if failures else 0

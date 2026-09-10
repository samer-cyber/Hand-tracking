"""Check the path from this PC to the phone, one link at a time.

Run when something is not connecting. Each check reports the specific fix rather
than a generic failure, because the four things that go wrong here -- no Apple
USB service, no cable, phone locked, app not started -- all present identically
as "it does not connect".
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


def main(port: int = 9871) -> int:
    print("HandPilot doctor\n")
    failures = 0

    print("Python")
    if sys.version_info < (3, 10):
        _bad(f"Python {sys.version.split()[0]} is too old",
             "install Python 3.10 or newer from python.org")
        failures += 1
    else:
        _ok(f"Python {sys.version.split()[0]}")
    try:
        import tkinter  # noqa: F401
        _ok("tkinter present")
    except ImportError:
        _bad("tkinter missing",
             "reinstall Python from python.org with the tcl/tk option ticked")
        failures += 1

    print("\nApple USB service")
    from handpilot import usbmux
    try:
        devices = usbmux.list_devices()
        _ok("usbmuxd is reachable on 127.0.0.1:27015")
    except usbmux.UsbmuxUnavailable as exc:
        _bad(str(exc))
        return failures + 1
    except Exception as exc:                                  # noqa: BLE001
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
        sock, device = usbmux.connect_first(port)
    except Exception as exc:                                  # noqa: BLE001
        _bad(str(exc),
             "open HandPilot on the phone and press Start, and check the port "
             "there matches")
        return failures + 1

    _ok(f"connected to {device['serial']}")
    sock.settimeout(3.0)
    start = time.perf_counter()
    try:
        data = sock.recv(65536)
    except socket.timeout:
        _bad("connected, but the phone is sending nothing",
             "press Start in the app; the camera row should read 'streaming'")
        sock.close()
        return failures + 1
    finally:
        pass

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
    if count >= 90:
        _ok(f"{count} messages/second")
    elif count > 0:
        _bad(f"only {count} messages/second",
             "expect 120 or so; check the frame rate and shutter on the Camera tab, "
             "and whether the room is bright enough for the shutter you chose")
        failures += 1
    else:
        _bad("no messages", "press Start in the phone app")
        failures += 1

    print("\nDone." if not failures else f"\n{failures} problem(s) found.")
    return 1 if failures else 0


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())

"""Talk to Apple's USB multiplexer, so a socket on this PC lands on a TCP port
inside the iPhone app.

Windows gets usbmuxd as part of Apple Mobile Device Support -- from iTunes, from
the Microsoft Store's "Apple Devices" app, or from the driver bundle Sideloadly
pulls in. All three end up listening on 127.0.0.1:27015 and speaking the same
plist protocol: ask it which devices are attached,
then ask it to ``Connect`` to a port on one of them, after which that same
socket becomes a raw pipe to whatever is listening inside the phone.

This is the same mechanism ``iproxy`` and Xcode use, reimplemented here in about
a hundred lines so that running the app is one command with nothing else to
install.

Note the direction: the mux can dial *into* the phone but the phone cannot dial
out through it. That is why HandPilot on the phone is the server and this side is
the client, which is the opposite of how the Wi-Fi version of this would
naturally be written.
"""

from __future__ import annotations

import plistlib
import socket
import struct
import subprocess
import sys

USBMUX_TCP_PORT = 27015
_HEADER = struct.Struct("<IIII")
_VERSION_PLIST = 1
_TYPE_PLIST = 8

# usbmuxd result codes. 3 is the one that actually happens in practice: the
# cable and the pairing are fine, the app just is not listening yet.
_RESULTS = {
    0: "ok",
    1: "usbmuxd rejected the request (bad command)",
    2: "no such device -- unplug and replug the phone",
    3: "connection refused -- HandPilot is not running on the phone, "
       "or its port does not match",
    5: "usbmuxd reported a bad version",
    6: "usbmuxd refused the connection (bad version)",
}


class UsbmuxError(RuntimeError):
    """Anything that went wrong between here and the phone."""


class UsbmuxUnavailable(UsbmuxError):
    """The multiplexer service itself could not be reached."""


def service_running() -> bool:
    """Whether Apple's device process is up.

    Two shapes exist and both end up on the same port. Classic iTunes installs a
    Windows service, ``Apple Mobile Device Service``; the Microsoft Store "Apple
    Devices" app ships the same thing as a per-user background process,
    ``AppleMobileDeviceProcess.exe``, which is only started when the app runs or
    a device is plugged in. That second case is the confusing one -- the software
    is installed, so nothing looks missing, and the port is simply not there.
    """
    if sys.platform != "win32":
        return True
    try:
        output = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq AppleMobileDeviceProcess.exe", "/NH"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "AppleMobileDeviceProcess" in output


def _no_service_message() -> str:
    if service_running():
        return ("Apple's device process is running but nothing is listening on "
                "127.0.0.1:27015. Unplug and replug the iPhone, or restart the "
                "Apple Devices app.")
    return ("Apple's device service is not running, so there is no USB tunnel to "
            "the phone. Plug the iPhone in and open the 'Apple Devices' app once "
            "-- that starts it and it then stays up. (No Apple software at all? "
            "Install Apple Devices from the Microsoft Store.) You can also switch "
            "this app to Wi-Fi on the Phone bar and skip the cable entirely.")


def _open_mux(timeout: float) -> socket.socket:
    if sys.platform == "win32":
        try:
            return socket.create_connection(("127.0.0.1", USBMUX_TCP_PORT), timeout=timeout)
        except OSError as exc:
            raise UsbmuxUnavailable(_no_service_message()) from exc
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect("/var/run/usbmuxd")
    except OSError as exc:
        sock.close()
        raise UsbmuxUnavailable("Cannot reach usbmuxd at /var/run/usbmuxd") from exc
    return sock


def _read_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    got = 0
    while got < count:
        chunk = sock.recv(count - got)
        if not chunk:
            raise UsbmuxError("usbmuxd closed the connection early")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _request(sock: socket.socket, payload: dict, tag: int) -> dict:
    payload = {
        "ClientVersionString": "handpilot",
        "ProgName": "handpilot",
        "kLibUSBMuxVersion": 3,
        **payload,
    }
    body = plistlib.dumps(payload)
    sock.sendall(_HEADER.pack(_HEADER.size + len(body), _VERSION_PLIST, _TYPE_PLIST, tag) + body)

    length, _version, _msgtype, _tag = _HEADER.unpack(_read_exactly(sock, _HEADER.size))
    if length < _HEADER.size or length > 1 << 22:
        raise UsbmuxError(f"usbmuxd sent an implausible reply length ({length})")
    return plistlib.loads(_read_exactly(sock, length - _HEADER.size))


def list_devices(timeout: float = 3.0) -> list[dict]:
    """Every device usbmuxd currently knows about, USB and Wi-Fi alike."""
    sock = _open_mux(timeout)
    try:
        reply = _request(sock, {"MessageType": "ListDevices"}, tag=1)
    finally:
        sock.close()

    out = []
    for entry in reply.get("DeviceList", []):
        props = entry.get("Properties", {})
        out.append(
            {
                "device_id": entry.get("DeviceID"),
                "serial": props.get("SerialNumber", "?"),
                # "Network" devices are reachable over Wi-Fi sync rather than the
                # cable, and carry none of the latency benefit we came for.
                "usb": props.get("ConnectionType") != "Network",
                "connection": props.get("ConnectionType", "USB"),
            }
        )
    return out


def first_usb_device(timeout: float = 3.0) -> dict:
    devices = list_devices(timeout)
    usb = [d for d in devices if d["usb"]]
    if not usb:
        if devices:
            raise UsbmuxError(
                "The phone is visible over Wi-Fi sync but not over the cable. "
                "Plug it in with a data-capable USB cable and unlock it."
            )
        raise UsbmuxError(
            "No iPhone attached. Plug it in, unlock it, and tap Trust if asked."
        )
    return usb[0]


def connect(device_id: int, port: int, timeout: float = 5.0) -> socket.socket:
    """Open a socket that terminates on ``port`` inside the phone."""
    sock = _open_mux(timeout)
    try:
        reply = _request(
            sock,
            {
                "MessageType": "Connect",
                "DeviceID": int(device_id),
                # usbmuxd wants this port in network byte order but typed as a
                # plain integer, so the swap has to happen here rather than in
                # the serialiser.
                "PortNumber": socket.htons(port),
            },
            tag=2,
        )
    except Exception:
        sock.close()
        raise

    number = reply.get("Number", -1)
    if number != 0:
        sock.close()
        raise UsbmuxError(_RESULTS.get(number, f"usbmuxd refused the connection (code {number})"))

    # From here the socket is a raw pipe, so the handshake timeout has to go --
    # a live stream that pauses while a hand is out of frame is not an error.
    sock.settimeout(None)
    return sock


def connect_first(port: int, timeout: float = 5.0) -> tuple[socket.socket, dict]:
    device = first_usb_device(timeout)
    return connect(device["device_id"], port, timeout), device

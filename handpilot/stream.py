"""Raw hand data from the phone, with nothing driving the mouse.

The pointer is one use of the stream. This module is for every other one:

    from handpilot import HandStream

    with HandStream() as hands:                 # USB; HandStream(wifi="192.168.1.20")
        for frame in hands:
            if frame.hand:
                x, y, confidence = frame.landmark("index_tip")
                print(f"index tip at {x:.3f}, {y:.3f}")

Each item is a :class:`handpilot.protocol.Frame`: 21 landmarks normalised to
0..1 (origin top-left, as the camera sees you -- not mirrored), confidences,
handedness, and the phone's capture timestamp. ``frame.to_dict()`` is the JSON
shape described in docs/data.md, which is also what the command line
(``python run.py --stream``) and the UDP forwarder emit.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time

from .link import Link
from .protocol import Frame


class HandStream:
    """Connect to the phone and iterate over frames as they arrive.

    By default only the newest frame is kept while you are busy, so a slow
    consumer always sees the hand where it is now rather than working through a
    backlog. Pass ``latest_only=False`` to receive every frame (for recording).
    """

    def __init__(self, wifi: str = "", port: int = 9871, latest_only: bool = True,
                 timeout: float | None = None) -> None:
        self.timeout = timeout
        self._queue: queue.Queue = queue.Queue(maxsize=1 if latest_only else 4096)
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._transport = "wifi" if wifi else "usb"
        self._host = wifi
        self._port = port
        self.link = Link(self._on_frame)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "HandStream":
        self.link.start(self._transport, self._host, self._port)
        return self

    def stop(self) -> None:
        self.link.stop()

    def __enter__(self) -> "HandStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ data

    def _on_frame(self, frame: Frame) -> None:
        with self._lock:
            self._latest = frame
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop the stale frame rather than the fresh one.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float | None = None) -> Frame | None:
        """The next frame, or None if none arrives within ``timeout`` seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def latest(self) -> Frame | None:
        """The most recent frame, without waiting -- handy inside a render loop."""
        with self._lock:
            return self._latest

    def __iter__(self):
        while True:
            frame = self.read(self.timeout)
            if frame is None:
                if self.timeout is not None:
                    return
                continue
            yield frame

    @property
    def connected(self) -> bool:
        return self.link.stats.connected

    @property
    def error(self) -> str:
        return self.link.stats.error


class UdpForwarder:
    """Send every frame as one JSON datagram, for programs in any language.

    One frame is about 1.3 KB, well inside a single UDP packet, and a datagram
    either arrives whole or not at all -- so a reader needs no framing, just
    ``recvfrom`` and a JSON parse. Sending never blocks the tracking thread.
    """

    def __init__(self, target: str = "127.0.0.1:9872") -> None:
        self.target = parse_target(target)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self.sent = 0
        self.error = ""

    def send(self, frame: Frame) -> None:
        try:
            self._sock.sendto(json.dumps(frame.to_dict(), separators=(",", ":")).encode(),
                              self.target)
            self.sent += 1
        except OSError as exc:
            # Nobody listening is normal for UDP on Windows (ICMP port
            # unreachable surfaces as an error on the next send); it is not a
            # reason to stop.
            self.error = str(exc)

    def close(self) -> None:
        self._sock.close()


def parse_target(text: str) -> tuple[str, int]:
    host, _, port = text.strip().rpartition(":")
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"expected host:port, got {text!r}")
    return host, int(port)


def write_stream(out, fmt: str = "json", wifi: str = "", port: int = 9871,
                 seconds: float | None = None) -> int:
    """The ``--stream`` command: frames to a text stream as JSON lines or CSV.

    Every frame is written, hand or not, so the timestamps stay continuous for
    anyone resampling or measuring gaps.
    """
    from .protocol import LANDMARK_NAMES

    count = 0
    deadline = time.monotonic() + seconds if seconds else None
    if fmt == "csv":
        columns = ["seq", "t", "hand", "handedness"]
        for name in LANDMARK_NAMES:
            columns += [f"{name}_x", f"{name}_y", f"{name}_c"]
        out.write(",".join(columns) + "\n")
    with HandStream(wifi=wifi, port=port, latest_only=False, timeout=0.5) as hands:
        try:
            while deadline is None or time.monotonic() < deadline:
                frame = hands.read(0.5)
                if frame is None:
                    continue
                row = frame.to_dict()
                if fmt == "csv":
                    cells = [str(row["seq"]), str(row["t"]), str(int(row["hand"])),
                             row["handedness"] or ""]
                    for point in (row["landmarks"] or [["", "", ""]] * len(LANDMARK_NAMES)):
                        cells += [str(v) for v in point]
                    out.write(",".join(cells) + "\n")
                else:
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                out.flush()
                count += 1
        except (KeyboardInterrupt, BrokenPipeError):
            # Ctrl+C, or the reader on the other end of a pipe went away.
            pass
    return count

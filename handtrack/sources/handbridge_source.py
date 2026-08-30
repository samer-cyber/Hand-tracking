"""Receiver for the HandBridge iPhone app.

The phone does its own hand detection and depth sampling, then sends ~384 bytes
per frame over UDP. That is the whole point: no video crosses the network, so
there is no encoder, no decoder, and no video buffering anywhere in the path.
Those were the largest unquantified terms in the latency budget when the phone
was acting as a webcam.

What arrives per frame:

  * 21 landmark positions in the phone's image, in pixels
  * a depth in metres at each landmark, or NaN where the sensor had nothing
  * per-landmark detector confidence
  * the camera's own intrinsic matrix, measured rather than assumed
  * two phone-clock timestamps, for latency accounting

The intrinsics matter more than they might appear. Every calibration attempt so
far has had to guess a field of view, and that guess feeds the essential-matrix
solve the entire rig geometry depends on. A wrong focal length there produces a
rig that looks self-consistent and is metrically wrong -- which is exactly the
failure we spent a long time chasing.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"HBR1"
LANDMARKS = 21
# 4 magic, 4 sequence, 8 capture time, 8 send time, 2 width, 2 height,
# 4 intrinsics floats, 4 status bytes, then 21 x (x, y, z, confidence).
PACKET = "<4sIddHH4f4B" + f"{LANDMARKS * 4}f"
PACKET_SIZE = struct.calcsize(PACKET)
DEFAULT_PORT = 9870


@dataclass
class HandBridgeFrame:
    """One frame of hand data from the phone."""
    pixels: np.ndarray          # (21,2) in the phone's image, pixels
    depth: np.ndarray           # (21,) metres, NaN where unmeasured
    confidence: np.ndarray      # (21,) 0..1
    intrinsics: np.ndarray      # 3x3 K for the phone image
    width: int
    height: int
    handedness: str             # "Left", "Right" or "Unknown"
    has_hand: bool
    timestamp: float            # local perf_counter at arrival
    phone_capture_time: float   # phone clock, when the frame was grabbed
    phone_send_time: float      # phone clock, just before transmit
    index: int                  # sequence number from the phone

    @property
    def image(self):
        """No image is transmitted; kept so callers can duck-type on frames."""
        return None


class HandBridgeSource:
    """Threaded UDP receiver exposing only the newest frame.

    Mirrors the camera sources' contract -- start(), read(), release() -- so the
    rest of the pipeline does not care where a view came from.
    """

    def __init__(self, port: int = DEFAULT_PORT, name: str = "phone"):
        self.port = int(port)
        self.name = name
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._latest: HandBridgeFrame | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        self._intervals: list[float] = []
        self._last_arrival: float | None = None
        self.packets = 0
        self.malformed = 0
        self.dropped = 0
        self._last_seq: int | None = None

        # Phone and PC clocks share no epoch. Tracking the smallest observed
        # (arrival - send) across the session estimates the offset: the fastest
        # packet is the one that spent least time in flight, so its difference
        # is the closest available approximation to the pure clock offset.
        self._min_offset: float | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self, timeout: float = 20.0, on_wait=None) -> "HandBridgeSource":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.settimeout(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="handbridge", daemon=True)
        self._thread.start()

        deadline = time.time() + timeout
        attempt = 0
        while self._latest is None and time.time() < deadline:
            time.sleep(0.05)
            attempt += 1
            if on_wait is not None and attempt % 20 == 0:
                on_wait(attempt // 20, max(0.0, deadline - time.time()))

        if self._latest is None:
            self.release()
            raise RuntimeError(
                f"No packets arrived on UDP port {self.port}.\n"
                "  1. Is HandBridge running on the phone with Start pressed?\n"
                "  2. Does the PC address in the app match this machine's LAN IP?\n"
                "     Run 'python tools/whats_my_ip.py' to see it.\n"
                "  3. Are both on the same network, and is the firewall allowing\n"
                f"     inbound UDP on port {self.port}? That is the usual culprit.")
        return self

    def _loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            frame = self._decode(data)
            if frame is None:
                self.malformed += 1
                continue

            now = frame.timestamp
            if self._last_arrival is not None:
                self._intervals.append(now - self._last_arrival)
                if len(self._intervals) > 120:
                    self._intervals.pop(0)
            self._last_arrival = now

            if self._last_seq is not None:
                gap = frame.index - self._last_seq
                if gap > 1:
                    self.dropped += gap - 1
            self._last_seq = frame.index

            offset = now - frame.phone_send_time
            if self._min_offset is None or offset < self._min_offset:
                self._min_offset = offset

            self.packets += 1
            with self._lock:
                self._latest = frame

    def _decode(self, data: bytes) -> HandBridgeFrame | None:
        if len(data) != PACKET_SIZE:
            return None
        try:
            fields = struct.unpack(PACKET, data)
        except struct.error:
            return None
        if fields[0] != MAGIC:
            return None

        seq = fields[1]
        capture_t, send_t = fields[2], fields[3]
        width, height = fields[4], fields[5]
        fx, fy, cx, cy = fields[6:10]
        chirality, has_hand, _has_depth, _pad = fields[10:14]
        block = np.array(fields[14:], dtype=np.float64).reshape(LANDMARKS, 4)

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return HandBridgeFrame(
            pixels=block[:, 0:2].copy(),
            depth=block[:, 2].copy(),
            confidence=block[:, 3].copy(),
            intrinsics=K,
            width=int(width), height=int(height),
            handedness={1: "Left", 2: "Right"}.get(chirality, "Unknown"),
            has_hand=bool(has_hand),
            timestamp=time.perf_counter(),
            phone_capture_time=capture_t,
            phone_send_time=send_t,
            index=int(seq),
        )

    # -- reading --------------------------------------------------------------

    def read(self) -> HandBridgeFrame | None:
        with self._lock:
            return self._latest

    @property
    def measured_fps(self) -> float:
        if len(self._intervals) < 5:
            return 0.0
        return 1.0 / float(np.median(self._intervals))

    @property
    def width(self) -> int:
        f = self.read()
        return 0 if f is None else f.width

    @property
    def height(self) -> int:
        f = self.read()
        return 0 if f is None else f.height

    def transit_ms(self, frame: HandBridgeFrame) -> float:
        """Time from the phone grabbing the frame to it arriving here.

        Only meaningful relative to the best packet seen so far, since the two
        clocks have no shared epoch. It measures phone-side processing plus
        network delay, which is the part of the budget this design exists to
        minimise, and it is honest about excluding what happens before the frame
        reaches the phone's own callback.
        """
        if self._min_offset is None:
            return float("nan")
        arrival_in_phone_clock = frame.timestamp - self._min_offset
        return (arrival_in_phone_clock - frame.phone_capture_time) * 1000.0

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def encode_packet(seq: int, capture_t: float, send_t: float, width: int, height: int,
                  fx: float, fy: float, cx: float, cy: float, chirality: int,
                  pixels: np.ndarray | None, depth: np.ndarray | None,
                  confidence: np.ndarray | None) -> bytes:
    """Build a packet exactly as the Swift sender does.

    Kept beside the decoder so the wire format has a single definition on this
    side, and so the layout can be tested without a phone present. A protocol
    whose two halves are only checked against each other on real hardware is a
    protocol that gets debugged at the worst possible moment.
    """
    block = np.full((LANDMARKS, 4), np.nan, dtype=np.float32)
    if pixels is not None:
        block[:, 0:2] = pixels
    if depth is not None:
        block[:, 2] = depth
    block[:, 3] = 0.0 if confidence is None else confidence
    return struct.pack(
        PACKET, MAGIC, seq, capture_t, send_t, width, height,
        fx, fy, cx, cy,
        chirality, 1 if pixels is not None else 0, 1 if depth is not None else 0, 0,
        *block.ravel().tolist())

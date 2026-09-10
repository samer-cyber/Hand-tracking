"""The wire format, desktop side. Mirror of ios/HandPilot/HandPilot/Wire.swift.

Every message is ``magic(4) type(1) length(2 LE)`` then ``length`` bytes. The
envelope exists because TCP is a byte stream: without it a single misaligned
byte would not lose one frame, it would silently reinterpret every landmark
after it as noise, and the symptom would be a hand that works for ten seconds
and then jitters forever.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

MAGIC = b"HPU1"
TYPE_FRAME = 1
TYPE_STATUS = 2
TYPE_PONG = 3

LANDMARKS = 21

_ENVELOPE = struct.Struct("<4sBH")
_FRAME_HEAD = struct.Struct("<IddBBBBHHf")
_POINTS = struct.Struct("<" + "fff" * LANDMARKS)

ORIENTATION_NAMES = {0: "portrait", 1: "landscape left", 2: "landscape right", 3: "upside down"}


@dataclass(slots=True)
class Frame:
    """One camera frame's landmarks, in oriented-image coordinates.

    ``x`` and ``y`` are normalised 0..1 with the origin at the top-left of the
    image as the user sees it -- the phone has already applied the mount rotation
    and flipped Vision's bottom-left convention, so nothing here needs to know
    how the phone was propped up.
    """

    seq: int = 0
    t_capture: float = 0.0
    t_send: float = 0.0
    hand: bool = False
    chirality: int = 0
    orientation: int = 0
    width: int = 0
    height: int = 0
    inference_ms: float = 0.0
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    conf: list = field(default_factory=list)
    #: Desktop clock, taken the moment the bytes came off the socket.
    t_recv: float = 0.0
    #: Capture-to-send inside the phone, in ms. Measurable without a shared clock
    #: because both ends of it are stamped on the phone's own clock.
    phone_ms: float = 0.0

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 16 / 9


def decode_frame(payload: bytes, t_recv: float) -> Frame:
    (seq, t_cap, t_send, hand, chirality, orientation, _pad,
     width, height, inference) = _FRAME_HEAD.unpack_from(payload, 0)
    values = _POINTS.unpack_from(payload, _FRAME_HEAD.size)
    return Frame(
        seq=seq,
        t_capture=t_cap,
        t_send=t_send,
        hand=bool(hand),
        chirality=chirality,
        orientation=orientation,
        width=width,
        height=height,
        inference_ms=inference,
        x=list(values[0::3]),
        y=list(values[1::3]),
        conf=list(values[2::3]),
        t_recv=t_recv,
        phone_ms=(t_send - t_cap) * 1000.0,
    )


class Reader:
    """Incremental parser. Feed it whatever ``recv`` returned; get back messages.

    Resynchronises on the magic rather than giving up, so a truncated stream
    costs one frame instead of the session.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes, t_recv: float) -> list[tuple[int, object]]:
        self._buf += data
        out: list[tuple[int, object]] = []
        while True:
            start = self._buf.find(MAGIC)
            if start < 0:
                # Keep only what could still be the front of a magic.
                if len(self._buf) > 3:
                    del self._buf[: len(self._buf) - 3]
                break
            if start:
                del self._buf[:start]
            if len(self._buf) < _ENVELOPE.size:
                break
            _magic, mtype, length = _ENVELOPE.unpack_from(self._buf, 0)
            total = _ENVELOPE.size + length
            if len(self._buf) < total:
                break
            payload = bytes(self._buf[_ENVELOPE.size:total])
            del self._buf[:total]

            if mtype == TYPE_FRAME and length >= _FRAME_HEAD.size + _POINTS.size:
                out.append((mtype, decode_frame(payload, t_recv)))
            elif mtype in (TYPE_STATUS, TYPE_PONG):
                try:
                    out.append((mtype, json.loads(payload.decode("utf-8"))))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
        return out


def encode_command(obj: dict) -> bytes:
    """PC to phone. Newline-delimited JSON, because this direction is a handful
    of settings changes a minute and readability is worth more than bytes."""
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

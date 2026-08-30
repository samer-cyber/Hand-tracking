"""Source factory, so callers can accept a webcam or the phone interchangeably.

A source spec is either an integer-like string (a camera index) or the literal
"record3d" for the iPhone's RGB+depth stream over USB. Both resulting objects
expose the same small contract -- read(), measured_fps, width, height,
release() -- and both return frames carrying .image, .timestamp and .index, so
calibration and runtime code never branches on which is which.
"""
from __future__ import annotations

from ..capture import CameraSource


def open_source(spec: str | int, width: int = 1280, height: int = 720,
                fps: int = 30, exposure: float | None = None, name: str = "cam"):
    """Open a capture source from a spec string."""
    text = str(spec).strip().lower()
    if text in ("record3d", "phone", "iphone"):
        from .record3d_source import Record3DSource

        def waiting(attempt: int, remaining: float) -> None:
            if attempt == 1:
                print("  waiting for Record3D to start streaming --")
                print("  open the app, enable USB Streaming mode, and press RECORD")
            print(f"  ...{remaining:.0f} s left", flush=True)

        return Record3DSource(name=name).start(timeout=45.0, on_wait=waiting)
    return CameraSource(int(text), width, height, fps, exposure=exposure, name=name).start()


def describe_spec(spec: str | int) -> str:
    text = str(spec).strip().lower()
    if text in ("record3d", "phone", "iphone"):
        return "iPhone via Record3D (RGB + TrueDepth)"
    return f"camera index {text}"

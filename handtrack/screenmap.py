"""Mapping a tracked hand onto screen coordinates, and detecting pinch clicks.

The mapping is learned rather than assumed. You point at each corner of the
screen in turn and the four correspondences define a homography from the plane
your hand moves in to the plane of the display. This matters for the
requirement that you can reach every part of the screen without stretching:
rather than imposing a fixed scale, the calibration adopts whatever region you
actually found comfortable, and the corners you chose become the screen edges.

A homography rather than a simple linear scale because your hand does not sweep
a neat rectangle parallel to the display. It moves through a region that is
rotated and foreshortened relative to the screen, and a homography is exactly
the transform that maps one quadrilateral onto another.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .detect import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST


class ScreenMapper:
    """Maps a hand position to a screen pixel, via a calibrated homography."""

    CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.H: np.ndarray | None = None
        self._samples: list[np.ndarray] = []

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def corner_targets(self, inset: float = 0.06) -> np.ndarray:
        """Screen points to aim at, pulled slightly inside the physical edges.

        Aiming exactly at a corner is awkward and tends to be done badly, which
        distorts the fit. Insetting a little produces a more reliable
        calibration, and the homography extrapolates to the true edges fine.
        """
        w, h = self.screen_width, self.screen_height
        ix, iy = w * inset, h * inset
        return np.array([[ix, iy], [w - ix, iy], [w - ix, h - iy], [ix, h - iy]], dtype=np.float64)

    def add_sample(self, hand_xy: np.ndarray) -> int:
        """Record the hand position for the next corner. Returns count so far."""
        self._samples.append(np.asarray(hand_xy, dtype=np.float64)[:2].copy())
        return len(self._samples)

    def undo_sample(self) -> int:
        if self._samples:
            self._samples.pop()
        return len(self._samples)

    def finish(self) -> bool:
        """Fit the homography once all four corners are recorded."""
        if len(self._samples) < 4:
            return False
        src = np.stack(self._samples[:4])
        dst = self.corner_targets()
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None or not np.all(np.isfinite(H)):
            return False
        self.H = H
        return True

    def reset(self) -> None:
        self._samples.clear()
        self.H = None

    @property
    def progress(self) -> int:
        return len(self._samples)

    def to_screen(self, hand_xy: np.ndarray) -> tuple[float, float] | None:
        """Map a hand position to screen pixels, clamped to the display."""
        if self.H is None:
            return None
        p = np.array([hand_xy[0], hand_xy[1], 1.0], dtype=np.float64)
        q = self.H @ p
        if abs(q[2]) < 1e-9:
            return None
        x, y = q[0] / q[2], q[1] / q[2]
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (float(np.clip(x, 0, self.screen_width - 1)),
                float(np.clip(y, 0, self.screen_height - 1)))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"H": self.H.tolist(),
                       "screen": [self.screen_width, self.screen_height]}, f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path) as f:
                d = json.load(f)
            self.H = np.array(d["H"], dtype=np.float64)
            return True
        except Exception:
            return False


class PinchDetector:
    """Detects a thumb-to-index pinch, scale- and distance-invariant.

    The raw gap between the fingertips is useless on its own, because it shrinks
    with distance from the camera exactly as a real pinch does. Dividing by the
    hand's own size makes the measure depend on the pose rather than on where
    the hand happens to be.

    Separate on and off thresholds give hysteresis. With a single threshold, a
    hand held near it produces a stream of alternating press and release events
    -- which is not a subtle degradation but an unusable interface.
    """

    def __init__(self, on_ratio: float = 0.34, off_ratio: float = 0.48,
                 min_frames: int = 2):
        self.on_ratio = float(on_ratio)
        self.off_ratio = float(off_ratio)
        self.min_frames = int(min_frames)
        self.pinching = False
        self.ratio = float("nan")
        self._streak = 0

    def update(self, points: np.ndarray) -> bool:
        """Feed a 21x3 hand pose. Returns the debounced pinch state."""
        if points is None or not np.all(np.isfinite(points[[THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP]])):
            self._streak = 0
            return self.pinching

        gap = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP]))
        scale = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))
        if scale < 1e-6:
            return self.pinching
        self.ratio = gap / scale

        want = self.ratio < self.on_ratio if not self.pinching else self.ratio < self.off_ratio
        if want != self.pinching:
            self._streak += 1
            if self._streak >= self.min_frames:
                self.pinching = want
                self._streak = 0
        else:
            self._streak = 0
        return self.pinching


class CursorStabiliser:
    """Extra smoothing applied only to the cursor.

    A cursor is judged far more harshly than a rendered skeleton: a wobble of a
    few pixels that is invisible on a drawn hand makes a pointer feel broken.
    The dead zone suppresses sub-pixel drift while the hand is essentially
    still, but is bypassed once real movement starts so that it never adds lag
    to a deliberate gesture.
    """

    def __init__(self, dead_zone_px: float = 2.5, smoothing: float = 0.35):
        self.dead_zone = float(dead_zone_px)
        self.smoothing = float(smoothing)
        self._pos: np.ndarray | None = None

    def reset(self) -> None:
        self._pos = None

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=np.float64)
        if self._pos is None:
            self._pos = p
            return float(p[0]), float(p[1])
        delta = p - self._pos
        dist = float(np.linalg.norm(delta))
        if dist < self.dead_zone:
            return float(self._pos[0]), float(self._pos[1])
        self._pos = self._pos + delta * (1.0 - self.smoothing)
        return float(self._pos[0]), float(self._pos[1])


class ClickAnchor:
    """Holds the cursor still while a pinch is forming.

    Pinching moves the index finger, so a cursor driven by the fingertip drifts
    at the exact moment you are trying to click, and the click lands somewhere
    other than where you aimed. Freezing the position briefly as the pinch
    begins makes clicks land where you pointed -- the difference between a demo
    and something you can actually use.
    """

    def __init__(self, freeze_seconds: float = 0.18):
        self.freeze_seconds = float(freeze_seconds)
        self._anchor: tuple[float, float] | None = None
        self._until = 0.0

    def apply(self, pos: tuple[float, float], pinch_started: bool,
              pinching: bool, now: float) -> tuple[float, float]:
        if pinch_started:
            self._anchor = pos
            self._until = now + self.freeze_seconds
        if self._anchor is not None and (pinching or now < self._until):
            return self._anchor
        self._anchor = None
        return pos

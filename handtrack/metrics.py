"""Instrumentation.

"Feels laggy" is not a number, and you cannot optimise what you cannot measure.
This module breaks the pipeline into its stages and reports where the
milliseconds actually go, plus a jitter figure measured while the hand is
deliberately still. Those two numbers -- motion-to-photon latency and
resting jitter -- are what separate tracking that feels attached to your hand
from tracking that feels like a laggy puppet.

Note on latency honesty: the timestamp we take is the moment `cap.read()`
returned, so everything before that (exposure integration, sensor readout, USB
transfer, driver buffering) is invisible to us and is NOT included. Expect a
further 10-30 ms of hidden latency upstream on a typical webcam. The
`tools/measure_true_latency.py` script measures the full loop optically to
recover that hidden portion.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np


class Rolling:
    """Fixed-window rolling statistics."""

    def __init__(self, size: int = 120):
        self.buf: deque[float] = deque(maxlen=size)

    def add(self, v: float) -> None:
        self.buf.append(float(v))

    @property
    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.buf)) if self.buf else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.buf, 95)) if self.buf else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.buf)) if self.buf else 0.0


class StageTimer:
    """Accumulates per-stage timings for one pass through the pipeline."""

    def __init__(self, size: int = 120):
        self.stages: dict[str, Rolling] = {}
        self.size = size
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        if self._t0 is not None:
            self.stages.setdefault(name, Rolling(self.size)).add((now - self._t0) * 1000.0)
        self._t0 = now

    def add(self, name: str, ms: float) -> None:
        self.stages.setdefault(name, Rolling(self.size)).add(ms)

    def summary(self) -> dict[str, float]:
        return {k: v.median for k, v in self.stages.items()}

    def total(self) -> float:
        return sum(v.median for v in self.stages.values())


class JitterMeter:
    """Measures residual positional noise while the hand is nearly still.

    Jitter measured during motion is meaningless (it is mostly real movement),
    so samples are only accumulated when the point is moving slowly. The result
    is the standard deviation of position in millimetres, which is directly
    comparable to published figures for commercial trackers.
    """

    def __init__(self, window: int = 90, still_threshold_mps: float = 0.05,
                 scale: float = 1000.0):
        # `scale` converts input units to display units: 1000 for metres->mm,
        # 1.0 when the input is already in pixels.
        self.window = window
        self.scale = float(scale)
        self.still_threshold = still_threshold_mps
        self._samples: deque[np.ndarray] = deque(maxlen=window)
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None
        self.is_still = False

    def update(self, point: np.ndarray, t: float) -> None:
        point = np.asarray(point, dtype=np.float64)
        if self._prev is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt > 1e-6:
                speed = float(np.linalg.norm(point - self._prev) / dt)
                self.is_still = speed < self.still_threshold
                if self.is_still:
                    self._samples.append(point.copy())
                else:
                    self._samples.clear()
        self._prev, self._prev_t = point.copy(), t

    @property
    def jitter(self) -> float:
        """RMS deviation from the mean position, in the configured units."""
        if len(self._samples) < 10:
            return float("nan")
        arr = np.stack(self._samples)
        dev = arr - arr.mean(0)
        return float(np.sqrt((dev ** 2).sum(1).mean()) * self.scale)

    # Retained for readability at call sites that work in millimetres.
    jitter_mm = jitter

    @property
    def samples(self) -> int:
        return len(self._samples)


class TrackingHealth:
    """Tracks detection continuity -- dropouts are felt as much as jitter."""

    def __init__(self, window: int = 300):
        self.hits: deque[int] = deque(maxlen=window)
        self.dropouts = 0
        self._was_tracked = False

    def update(self, tracked: bool) -> None:
        self.hits.append(1 if tracked else 0)
        if self._was_tracked and not tracked:
            self.dropouts += 1
        self._was_tracked = tracked

    @property
    def rate(self) -> float:
        return float(np.mean(self.hits)) if self.hits else 0.0

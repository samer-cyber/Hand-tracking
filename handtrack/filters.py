"""Temporal filters for landmark streams.

The naive fix for jitter is a heavy low-pass filter, which is exactly what
makes cheap hand tracking feel like it is being dragged through mud: constant
smoothing buys steadiness when the hand is still at the price of lag when it
moves, and the hand moving is precisely when you care. The One Euro filter
resolves this by making the cutoff frequency a function of speed -- smooth hard
when slow (killing jitter you would otherwise see), barely smooth at all when
fast (so motion stays crisp).

On top of that we run a deliberate forward prediction. Every stage of the
pipeline (exposure, USB transfer, inference, compositing) costs real
milliseconds that no filter can undo, so instead we extrapolate along the
smoothed velocity to land the estimate where the hand will be when the frame is
actually shown. This is the same trick head-mounted displays use, and it is the
single largest contributor to a system feeling "attached" to your hand.
"""
from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Speed-adaptive low-pass filter, vectorised over an arbitrary array shape.

    min_cutoff sets the smoothing floor when stationary (lower = steadier but
    laggier). beta sets how aggressively the cutoff opens up with speed
    (higher = less lag when moving, more jitter passed through).
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        # Guard against duplicate timestamps and clock hiccups; a bad dt here
        # produces a spike that looks exactly like a tracking failure.
        if dt <= 1e-6:
            return self._x_prev.copy()
        dt = float(np.clip(dt, 1e-3, 0.2))

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha_array(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        self.velocity = dx_hat
        return x_hat


def _alpha_array(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class VelocityPredictor:
    """Extrapolates forward to cancel measured end-to-end pipeline latency.

    Prediction is a noise amplifier, so this is deliberately conservative: the
    velocity is smoothed before use, the lead time is capped, and the resulting
    displacement is clamped. Over-prediction reads as overshoot and wobble,
    which feels worse than the lag it was meant to fix.
    """

    def __init__(self, lead_seconds: float = 0.030, max_step: float = 0.06, smoothing: float = 0.6):
        self.lead_seconds = float(lead_seconds)
        self.max_step = float(max_step)
        self.smoothing = float(smoothing)
        self._vel: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._vel = self._prev = self._t_prev = None

    def __call__(self, x: np.ndarray, t: float, lead: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        lead = self.lead_seconds if lead is None else float(lead)

        if self._prev is None:
            self._prev, self._t_prev = x.copy(), t
            self._vel = np.zeros_like(x)
            return x.copy()

        dt = t - self._t_prev
        if dt > 1e-6:
            inst = (x - self._prev) / max(dt, 1e-3)
            self._vel = self.smoothing * self._vel + (1.0 - self.smoothing) * inst
            self._prev, self._t_prev = x.copy(), t

        step = np.clip(self._vel * lead, -self.max_step, self.max_step)
        return x + step


class ConfidenceGate:
    """Suppresses the pop that occurs when tracking briefly drops and recovers.

    A detector that loses the hand for two frames and reacquires it slightly
    offset produces a visible jump. Rather than trusting the first frame after a
    gap, we blend back in over a few frames.
    """

    def __init__(self, reacquire_frames: int = 4):
        self.reacquire_frames = int(reacquire_frames)
        self._since_gap = 0
        self._last_good: np.ndarray | None = None

    def __call__(self, x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            self._since_gap = 0
            return None
        x = np.asarray(x, dtype=np.float64)
        if self._last_good is None or self._since_gap >= self.reacquire_frames:
            self._since_gap += 1
            self._last_good = x.copy()
            return x
        self._since_gap += 1
        w = self._since_gap / float(self.reacquire_frames)
        blended = w * x + (1.0 - w) * self._last_good
        self._last_good = blended
        return blended

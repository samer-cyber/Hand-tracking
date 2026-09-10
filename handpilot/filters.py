"""Smoothing that does not cost lag when it matters.

A plain low-pass has one setting and two failure modes: enough smoothing to hold
the cursor still on a stationary hand always adds enough lag to make a fast
gesture feel like dragging the pointer through syrup, and vice versa. The One
Euro filter [Casiez, Roussel & Vogel, CHI 2012] fixes this by making the cutoff
frequency a function of speed -- heavy smoothing while the hand is nearly still,
almost none while it is moving fast, and the eye only notices jitter in the first
case and only notices lag in the second.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro2D:
    """One Euro over a 2D point, with a single speed shared by both axes.

    Sharing matters: filtering x and y independently gives them different
    effective cutoffs during any diagonal movement, so a straight diagonal swipe
    comes out visibly bent.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = 0.0
        self._y = 0.0
        self._dx = 0.0
        self._dy = 0.0
        self._t = 0.0
        self._primed = False

    def reset(self) -> None:
        self._primed = False
        self._dx = self._dy = 0.0

    @property
    def velocity(self) -> tuple[float, float]:
        """Filtered units per second -- reused for the small lead-ahead term."""
        return (self._dx, self._dy)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        if not self._primed:
            self._x, self._y, self._t = x, y, t
            self._primed = True
            return x, y

        dt = t - self._t
        if dt <= 0:
            return self._x, self._y
        self._t = t

        a_d = _alpha(self.d_cutoff, dt)
        self._dx += a_d * ((x - self._x) / dt - self._dx)
        self._dy += a_d * ((y - self._y) / dt - self._dy)

        speed = math.hypot(self._dx, self._dy)
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self._x += a * (x - self._x)
        self._y += a * (y - self._y)
        return self._x, self._y


class Hysteresis:
    """A latch with separate on and off thresholds.

    Every pinch, curl and contact test in this project runs through one of these.
    A single threshold on a noisy signal does not produce "a click" -- it produces
    a burst of clicks while the measurement sits on the boundary, which is the
    single most common way hand-tracked input feels broken.
    """

    def __init__(self, on_below: float, off_above: float) -> None:
        self.on_below = on_below
        self.off_above = off_above
        self.state = False

    def update(self, value: float) -> bool:
        if self.state:
            if value > self.off_above:
                self.state = False
        elif value < self.on_below:
            self.state = True
        return self.state

    def reset(self) -> None:
        self.state = False

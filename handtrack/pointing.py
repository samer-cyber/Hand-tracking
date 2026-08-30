"""Turning a tracked hand into a screen position.

There is no single right answer here, so all the plausible schemes are
implemented and switchable at runtime. What follows is what the geometry says
about each, because the differences are large and not obvious from trying them
for ten seconds.

The governing number is the lever arm. Pointing is angular: an error of one
degree in the direction you are aiming lands the cursor further from the target
the further away the screen is. Measured on a 15.5-inch screen at 40 cm, one
degree of aim error moves the cursor about 75 pixels. So the question for any
ray-based mode is not "how accurately can the hand be tracked" -- millimetres,
easily -- but "how accurately can a person aim a direction", which is more like
one to three degrees.

Measured on a 15.5-inch screen at 40 cm, with 2 mm of tracking jitter
(tools/test_pointing.py):

    mode          cursor jitter   hand travel to cross the screen
    eye ray            28 px                  36 cm
    finger ray        100 px                  39 cm
    absolute            7 px                  47 cm
    relative            8 px                  72 cm

Two things fall out of that, and the second contradicts the obvious expectation.

Among rays, the ORIGIN dominates. A ray from the knuckle to the fingertip has a
lever of about 10 cm, so finger wobble becomes a large angle; a ray from the
head through the fingertip has a lever of about 35 cm and is three and a half
times steadier for identical tracking. That is the geometry behind gaze-plus-
pinch on Vision Pro, approximated here with an assumed head position since we
cannot see the eyes.

But *any* ray is beaten by absolute positioning, by a factor of four. A ray
amplifies positional jitter by roughly (distance to screen) / (lever arm), and
since the screen is further away than your hand that factor is always above one.
Absolute mapping has no such term: a millimetre of hand jitter is a millimetre
of cursor jitter, scaled only by the region-to-screen ratio.

This is worth stating plainly because published comparisons generally find
ray-casting superior to direct pointing, and that is not wrong -- it is measured
in VR, where targets are metres away and physically out of reach, so absolute
mapping is not an option. At a desk the screen is within arm's length, a
comfortable hand region maps cleanly onto it, and the geometry reverses.

The trade absolute makes is reach: it requires holding your arm inside a
particular volume, whereas a ray works from wherever your hand happens to be.
Relative mode is the compromise -- nearly as steady as absolute, works from any
hand position, but gives up the direct spatial correspondence and needs the most
travel unless the gain is raised.
"""
from __future__ import annotations

import numpy as np

from .detect import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, WRIST
from .screen3d import ScreenGeometry

MODE_EYE_RAY = "eye ray"
MODE_FINGER_RAY = "finger ray"
MODE_ABSOLUTE = "absolute"
MODE_RELATIVE = "relative"
MODES = [MODE_EYE_RAY, MODE_FINGER_RAY, MODE_ABSOLUTE, MODE_RELATIVE]

MODE_HELP = {
    MODE_EYE_RAY: "point at the screen; ray runs from your head through your fingertip",
    MODE_FINGER_RAY: "point at the screen; ray runs along your finger only",
    MODE_ABSOLUTE: "hand position maps directly onto the screen, like reaching for it",
    MODE_RELATIVE: "hand motion moves the cursor, like a trackpad with gain",
}


def finger_ray(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Origin and direction along the index finger.

    Anchored at the knuckle rather than the wrist: the wrist-to-tip vector
    swings whenever the finger curls, so a pinch would drag the cursor at the
    exact moment of clicking.
    """
    if points is None or not np.all(np.isfinite(points[[INDEX_MCP, INDEX_TIP]])):
        return None
    o = points[INDEX_MCP]
    d = points[INDEX_TIP] - o
    if np.linalg.norm(d) < 1e-4:
        return None
    return o, d


def eye_ray(points: np.ndarray, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Ray from an assumed head position through the fingertip.

    The long lever is the entire point: the same fingertip jitter subtends a
    much smaller angle from 60 cm away than from the 10 cm of the finger itself,
    so the cursor is dramatically steadier for identical tracking quality.
    """
    if points is None or not np.all(np.isfinite(points[INDEX_TIP])):
        return None
    tip = points[INDEX_TIP]
    d = tip - eye
    if np.linalg.norm(d) < 1e-3:
        return None
    return eye, d


def hand_anchor(points: np.ndarray) -> np.ndarray | None:
    """A stable point on the hand for position-based modes.

    The midpoint of the knuckles, not a fingertip. Fingertips move when you
    gesture; the knuckles move only when the hand does, so the cursor stays put
    while you pinch.
    """
    if points is None or not np.all(np.isfinite(points[[INDEX_MCP, MIDDLE_MCP, WRIST]])):
        return None
    return (points[INDEX_MCP] + points[MIDDLE_MCP]) / 2.0


class AbsoluteRegion:
    """Maps a calibrated box of hand positions onto the screen.

    The box is learned from where your hand actually went during calibration,
    then padded, so the reachable region is the one you found comfortable rather
    than one imposed on you.
    """

    def __init__(self, margin: float = 0.12):
        self.margin = float(margin)
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def fit(self, samples: np.ndarray) -> bool:
        s = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        if len(s) < 4:
            return False
        lo, hi = s.min(0), s.max(0)
        span = np.maximum(hi - lo, 0.05)
        self.lo = lo - span * self.margin
        self.hi = hi + span * self.margin
        return True

    @property
    def calibrated(self) -> bool:
        return self.lo is not None

    def to_screen(self, point: np.ndarray, geo: ScreenGeometry) -> tuple[float, float] | None:
        if not self.calibrated or not geo.calibrated:
            return None
        span = np.maximum(self.hi - self.lo, 1e-6)
        n = (np.asarray(point, dtype=np.float64) - self.lo) / span
        # Screen x follows hand x; screen y is inverted because the screen's
        # vertical axis points downward while the world's points up.
        return geo._to_pixels(float(n[0]), float(1.0 - n[1]))


class RelativePointer:
    """Trackpad-style mapping: motion drives the cursor, with gain.

    Acceleration is included for the same reason desktop mice have it: a single
    fixed gain forces a choice between crossing the screen quickly and landing
    on small targets. Scaling gain with speed gives both -- fast movements cover
    distance, slow ones refine.
    """

    def __init__(self, gain: float = 2600.0, accel: float = 1.6,
                 screen: tuple[int, int] = (1920, 1080)):
        self.gain = float(gain)
        self.accel = float(accel)
        self.screen = screen
        self.x = screen[0] / 2.0
        self.y = screen[1] / 2.0
        self._prev: np.ndarray | None = None

    def reset(self, centre: bool = False) -> None:
        self._prev = None
        if centre:
            self.x, self.y = self.screen[0] / 2.0, self.screen[1] / 2.0

    def update(self, point: np.ndarray | None, dt: float) -> tuple[float, float]:
        if point is None or not np.all(np.isfinite(point)):
            self._prev = None
            return self.x, self.y
        p = np.asarray(point, dtype=np.float64)
        if self._prev is None:
            self._prev = p
            return self.x, self.y

        delta = p - self._prev
        self._prev = p
        speed = float(np.linalg.norm(delta)) / max(dt, 1e-3)
        g = self.gain * (1.0 + self.accel * min(speed, 2.0))
        self.x = float(np.clip(self.x + delta[0] * g, 0, self.screen[0] - 1))
        self.y = float(np.clip(self.y - delta[1] * g, 0, self.screen[1] - 1))
        return self.x, self.y


class Pointer:
    """Applies whichever pointing mode is active."""

    def __init__(self, geo: ScreenGeometry, screen: tuple[int, int]):
        self.geo = geo
        self.screen = screen
        self.mode = MODE_EYE_RAY
        self.region = AbsoluteRegion()
        self.relative = RelativePointer(screen=screen)
        # Assumed head position, used by the eye-ray mode. Refined at
        # calibration time from where the hand actually was, on the reasoning
        # that people point roughly outward from in front of their face.
        self.eye = np.array([0.0, 0.10, -0.25])
        self.last_ray: tuple[np.ndarray, np.ndarray] | None = None

    def set_mode(self, mode: str) -> None:
        if mode in MODES:
            self.mode = mode
            self.relative.reset()

    def cycle(self) -> str:
        self.set_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])
        return self.mode

    def estimate_eye(self, hand_samples: np.ndarray) -> None:
        """Place the assumed head behind and above the calibration region.

        A rough estimate is enough. What the eye ray needs is a long lever arm,
        and being 10 cm off along it changes the pointing direction very little
        compared with the couple of degrees of human aiming error it is there to
        suppress.
        """
        s = np.asarray(hand_samples, dtype=np.float64).reshape(-1, 3)
        if len(s) < 4:
            return
        centre = s.mean(0)
        # Behind the hand relative to the screen, and a little above it.
        away = centre - self.geo.origin if self.geo.calibrated else np.array([0, 0, -1.0])
        away = away / max(np.linalg.norm(away), 1e-6)
        self.eye = centre + away * 0.35 + np.array([0.0, 0.12, 0.0])

    def update(self, points: np.ndarray | None, dt: float) -> tuple[float, float] | None:
        self.last_ray = None
        if points is None:
            if self.mode == MODE_RELATIVE:
                return self.relative.update(None, dt)
            return None

        if self.mode == MODE_EYE_RAY:
            ray = eye_ray(points, self.eye)
            if ray is None:
                return None
            self.last_ray = ray
            return self.geo.intersect_ray(*ray)

        if self.mode == MODE_FINGER_RAY:
            ray = finger_ray(points)
            if ray is None:
                return None
            self.last_ray = ray
            return self.geo.intersect_ray(*ray)

        if self.mode == MODE_ABSOLUTE:
            anchor = hand_anchor(points)
            if anchor is None:
                return None
            return self.region.to_screen(anchor, self.geo)

        return self.relative.update(hand_anchor(points), dt)

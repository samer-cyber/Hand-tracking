"""Pose and motion recognition, and the state machine that decides when a
recognised pose becomes a trigger.

Two ideas carry most of the weight here.

**Exactly one pose is active at a time.** The predicates below overlap -- an OK
sign is also a pinch with three fingers out, a fist is a thumbs-up whose thumb
happens to point sideways. Evaluating them independently means a hand passing
through a shape on its way to another fires whatever it brushed. So they are
ranked and the first match wins, most specific first.

**Nothing fires on a single frame.** At 120 Hz a pose held for a tenth of a
second is twelve frames, and every real gesture passes through several
accidental shapes on the way in and out. A hold requirement filters those out for
free, and a cooldown stops a held pose from repeating unless it was asked to.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .handmodel import Hand

CURLED = 0.30
EXTENDED = 0.62

THUMB, INDEX, MIDDLE, RING, LITTLE = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class GestureSpec:
    id: str
    label: str
    hint: str
    dynamic: bool = False


#: Order is priority order for the static poses: the first one that matches wins.
GESTURES: tuple[GestureSpec, ...] = (
    GestureSpec("ok_sign", "OK sign", "Thumb and index touching, other three out"),
    GestureSpec("pinch", "Pinch", "Thumb and index touching, hand otherwise closed"),
    GestureSpec("thumbs_up", "Thumbs up", "Thumb up, everything else curled"),
    GestureSpec("thumbs_down", "Thumbs down", "Thumb down, everything else curled"),
    GestureSpec("shaka", "Shaka", "Thumb and little finger out"),
    GestureSpec("rock", "Rock horns", "Index and little finger out"),
    GestureSpec("peace", "Peace sign", "Index and middle out"),
    GestureSpec("three", "Three fingers", "Index, middle and ring out"),
    GestureSpec("point", "Pointing", "Index only"),
    GestureSpec("fist", "Fist", "Everything curled"),
    GestureSpec("open_palm", "Open palm", "All five fingers spread"),
    GestureSpec("swipe_left", "Swipe left", "Move the hand left, quickly", dynamic=True),
    GestureSpec("swipe_right", "Swipe right", "Move the hand right, quickly", dynamic=True),
    GestureSpec("swipe_up", "Swipe up", "Move the hand up, quickly", dynamic=True),
    GestureSpec("swipe_down", "Swipe down", "Move the hand down, quickly", dynamic=True),
)

GESTURE_IDS = tuple(g.id for g in GESTURES)
GESTURE_BY_ID = {g.id: g for g in GESTURES}


def _curled(v: float) -> bool:
    return v < CURLED


def _out(v: float) -> bool:
    return v > EXTENDED


def classify_pose(h: Hand) -> str | None:
    """The one static pose this hand is currently making, if any."""
    if not h.ok:
        return None
    t, i, m, r, l = h.ext
    pinched = h.pinch_index < 0.34

    if pinched and _out(m) and _out(r) and _out(l):
        return "ok_sign"
    if pinched and not _out(m) and not _out(r):
        return "pinch"
    if _curled(i) and _curled(m) and _curled(r) and _curled(l) and _out(t):
        # The thumb has to actually point somewhere for up and down to mean
        # anything; a thumb lying across the fingers is just a fist.
        dy = h.thumb_dir[1]
        if dy < -0.55:
            return "thumbs_up"
        if dy > 0.55:
            return "thumbs_down"
        return "fist"
    if _out(t) and _out(l) and _curled(i) and _curled(m) and _curled(r):
        return "shaka"
    if _out(i) and _out(l) and _curled(m) and _curled(r):
        return "rock"
    if _out(i) and _out(m) and _out(r) and not _out(l):
        return "three"
    if _out(i) and _out(m) and _curled(r) and _curled(l):
        return "peace"
    if _out(i) and _curled(m) and _curled(r) and _curled(l):
        return "point"
    if _curled(i) and _curled(m) and _curled(r) and _curled(l):
        return "fist"
    if _out(t) and _out(i) and _out(m) and _out(r) and _out(l):
        return "open_palm"
    return None


@dataclass
class SwipeConfig:
    #: Window over which displacement is measured.
    window_ms: float = 260.0
    #: Minimum travel, in palm widths -- so it does not depend on distance.
    min_travel: float = 1.7
    #: How much the winning axis must beat the other, so a diagonal flail does
    #: not count as both a horizontal and a vertical swipe.
    dominance: float = 1.9
    #: Require an open-ish hand, which keeps swipes from firing during the large
    #: hand movements that happen while doing something else entirely.
    require_open: bool = True
    cooldown_ms: float = 700.0


class SwipeDetector:
    def __init__(self, config: SwipeConfig | None = None) -> None:
        self.config = config or SwipeConfig()
        self._trail: deque = deque(maxlen=256)
        self._last_fire = 0.0

    def reset(self) -> None:
        self._trail.clear()

    def update(self, h: Hand, now: float, flip_x: bool, flip_y: bool) -> str | None:
        if not h.ok:
            self._trail.clear()
            return None

        self._trail.append((now, h.palm_x, h.palm_y, h.palm, sum(h.ext[1:])))
        window = self.config.window_ms / 1000.0
        while self._trail and now - self._trail[0][0] > window:
            self._trail.popleft()
        if len(self._trail) < 4:
            return None
        if (now - self._last_fire) * 1000.0 < self.config.cooldown_ms:
            return None

        t0, x0, y0, palm0, _ = self._trail[0]
        _, x1, y1, palm1, open1 = self._trail[-1]
        if self.config.require_open and open1 < 2.4:
            return None

        palm = max((palm0 + palm1) * 0.5, 1e-4)
        # Displacement in palm widths. Both the centre and the palm come from
        # the aspect-corrected coordinates, so a sideways swipe and a vertical
        # one of the same physical size measure the same -- which is what makes
        # the dominance test below mean anything.
        dx = (x1 - x0) / palm
        dy = (y1 - y0) / palm
        if flip_x:
            dx = -dx
        if flip_y:
            dy = -dy

        adx, ady = abs(dx), abs(dy)
        if adx > ady * self.config.dominance and adx > self.config.min_travel:
            self._last_fire = now
            self._trail.clear()
            return "swipe_right" if dx > 0 else "swipe_left"
        if ady > adx * self.config.dominance and ady > self.config.min_travel:
            self._last_fire = now
            self._trail.clear()
            # y grows downward in image coordinates.
            return "swipe_down" if dy > 0 else "swipe_up"
        return None


@dataclass
class Binding:
    enabled: bool = False
    action: str = "none"
    #: "once" fires on the leading edge; "repeat" keeps firing while held, which
    #: is what volume and scrolling actually want.
    mode: str = "once"
    hold_ms: float = 180.0
    repeat_ms: float = 200.0
    param: str = ""


@dataclass
class _PoseState:
    since: float = 0.0
    fired: bool = False
    last_repeat: float = 0.0


class GestureEngine:
    """Poses in, (gesture_id, binding) pairs out."""

    def __init__(self) -> None:
        self.bindings: dict[str, Binding] = {gid: Binding() for gid in GESTURE_IDS}
        self.swipes = SwipeDetector()
        self.cooldown_ms = 450.0
        self._states: dict[str, _PoseState] = {gid: _PoseState() for gid in GESTURE_IDS}
        self._current: str | None = None
        self._last_fire: dict[str, float] = {}
        #: What the UI shows -- the pose being held right now, fired or not.
        self.active_pose: str | None = None

    def reset(self) -> None:
        self._current = None
        self.active_pose = None
        self.swipes.reset()
        for state in self._states.values():
            state.since = 0.0
            state.fired = False

    def update(self, h: Hand, now: float, flip_x: bool = False,
               flip_y: bool = False) -> list[tuple[str, Binding]]:
        fired: list[tuple[str, Binding]] = []

        swipe = self.swipes.update(h, now, flip_x, flip_y)
        if swipe is not None:
            binding = self.bindings.get(swipe)
            if binding and binding.enabled and binding.action != "none":
                if (now - self._last_fire.get(swipe, 0.0)) * 1000.0 >= self.cooldown_ms:
                    self._last_fire[swipe] = now
                    fired.append((swipe, binding))

        pose = classify_pose(h)
        self.active_pose = pose
        if pose != self._current:
            if self._current is not None:
                state = self._states[self._current]
                state.since = 0.0
                state.fired = False
            self._current = pose
            if pose is not None:
                self._states[pose].since = now
                self._states[pose].fired = False

        if pose is None:
            return fired

        binding = self.bindings.get(pose)
        if not binding or not binding.enabled or binding.action == "none":
            return fired

        state = self._states[pose]
        held_ms = (now - state.since) * 1000.0
        if held_ms < binding.hold_ms:
            return fired

        if not state.fired:
            if (now - self._last_fire.get(pose, 0.0)) * 1000.0 >= self.cooldown_ms:
                state.fired = True
                state.last_repeat = now
                self._last_fire[pose] = now
                fired.append((pose, binding))
        elif binding.mode == "repeat":
            if (now - state.last_repeat) * 1000.0 >= binding.repeat_ms:
                state.last_repeat = now
                fired.append((pose, binding))
        return fired

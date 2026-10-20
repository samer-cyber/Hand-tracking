"""Turning 21 raw landmarks into the handful of numbers everything else uses.

Two corrections happen here and both are load-bearing:

**Aspect.** Landmarks arrive normalised 0..1 against a 16:9 image, so a
horizontal distance of 0.1 is nearly twice a vertical one. Comparing them
directly makes every pinch threshold depend on which way the fingers happen to
be pointing -- pinching horizontally would register at a different gap than
pinching vertically. Everything below works in units of image *height*.

**Scale.** The hand is bigger in frame when it is closer, so no distance is
meaningful in absolute terms. Every measurement is divided by the palm, which
makes the whole feature set independent of how far away you are sitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
LITTLE_MCP, LITTLE_PIP, LITTLE_DIP, LITTLE_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, LITTLE_TIP)
MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)

#: The knuckle row the pointer rides on by default. Knuckles are the steadiest
#: points a hand model reports -- fingertips sit at the end of a long chain of
#: guesses and the thumb tip is the noisiest of all -- and they barely move when
#: the fingers pinch, which is exactly when the pointer must not move.
KNUCKLES = (INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)

#: Drawn by the desktop preview, and a useful sanity check when tracking looks
#: wrong: a skeleton with crossed bones means the landmark order is off.
BONES = (
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, LITTLE_MCP),
    (LITTLE_MCP, LITTLE_PIP), (LITTLE_PIP, LITTLE_DIP), (LITTLE_DIP, LITTLE_TIP),
    (WRIST, LITTLE_MCP),
)

#: How far a fingertip sits from its own knuckle, in palm units, when the finger
#: is fully curled versus fully straight. Measured from the two extremes rather
#: than tuned by feel, and used to turn a raw distance into a 0..1 extension.
#: The thumb gets its own pair because it folds across the palm rather than into
#: it, so it never gets as close to its knuckle as the fingers do.
_CURL_RANGE = {
    "thumb": (0.45, 0.95),
    "index": (0.40, 1.05),
    "middle": (0.42, 1.10),
    "ring": (0.40, 1.02),
    "little": (0.36, 0.88),
}


@dataclass(slots=True)
class Hand:
    """Aspect-corrected, palm-normalised features for one frame."""

    ok: bool
    #: Aspect-corrected coordinates in units of image height, origin top-left.
    px: list
    py: list
    conf: list
    palm: float
    #: 0 curled .. 1 straight, per finger, in FINGER_NAMES order.
    ext: tuple
    #: Fingertip-to-thumbtip gaps in palm units.
    pinch_index: float
    pinch_middle: float
    pinch_ring: float
    pinch_little: float
    #: Midpoint of thumb and index tips -- the pointer's anchor, in raw
    #: normalised 0..1 image coordinates rather than aspect-corrected ones,
    #: because that is what maps onto a screen.
    point_x: float
    point_y: float
    #: Palm centre, in aspect-corrected units of image height -- *not* the raw
    #: normalised coordinates the pointer uses. It feeds the swipe detector,
    #: which divides displacement by ``palm``, and ``palm`` is measured in those
    #: same units. Mixing the two silently scales horizontal travel by the aspect
    #: ratio, so a sideways swipe would need nearly twice the distance of an
    #: identical vertical one and diagonals would be misclassified.
    palm_x: float
    palm_y: float
    #: Unit vector from thumb knuckle to thumb tip, aspect-corrected. y is
    #: positive downwards, matching the image.
    thumb_dir: tuple
    #: Mean confidence over the joints any decision here depends on.
    quality: float
    #: Confidence-weighted centre of the knuckle row, in the same raw normalised
    #: coordinates as ``point_x``/``point_y``.
    knuckle_x: float = 0.0
    knuckle_y: float = 0.0

    def anchor(self, kind: str = "knuckles") -> tuple[float, float]:
        """The point the cursor follows, in normalised image coordinates."""
        if kind == "pinch":
            return self.point_x, self.point_y
        return self.knuckle_x, self.knuckle_y


def _dist(px, py, a: int, b: int) -> float:
    return math.hypot(px[a] - px[b], py[a] - py[b])


def build(frame, conf_min: float = 0.3) -> Hand:
    """Derive features, or return ``ok=False`` if the frame cannot support them."""
    empty = Hand(False, [], [], [], 0.0, (0.0,) * 5, 1.0, 1.0, 1.0, 1.0,
                 0.0, 0.0, 0.0, 0.0, (0.0, -1.0), 0.0)
    if not frame.hand or len(frame.x) < 21:
        return empty

    aspect = frame.aspect
    px = [v * aspect for v in frame.x]
    py = list(frame.y)
    conf = frame.conf

    # The wrist and the knuckle row define the palm, and the palm defines every
    # other measurement -- so if those specific points are unreliable, nothing
    # downstream can be trusted no matter how confident the fingertips look.
    core = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)
    quality = sum(conf[i] for i in core) / len(core)
    if quality < conf_min:
        return empty

    # Two spans rather than one: palm length collapses when the hand is seen
    # edge-on and knuckle width collapses when it is seen end-on, but they do not
    # collapse together, so the larger of the two survives both.
    palm = max(_dist(px, py, WRIST, MIDDLE_MCP), _dist(px, py, INDEX_MCP, LITTLE_MCP))
    if palm < 1e-4:
        return empty

    ext = []
    for name, tip, mcp in zip(FINGER_NAMES, TIPS, MCPS):
        lo, hi = _CURL_RANGE[name]
        reach = _dist(px, py, tip, mcp) / palm
        ext.append(min(1.0, max(0.0, (reach - lo) / (hi - lo))))

    def gap(tip: int) -> float:
        return _dist(px, py, THUMB_TIP, tip) / palm

    tdx = px[THUMB_TIP] - px[THUMB_MCP]
    tdy = py[THUMB_TIP] - py[THUMB_MCP]
    tlen = math.hypot(tdx, tdy) or 1.0

    palm_ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, LITTLE_MCP)
    weights = [max(conf[i], 1e-3) for i in KNUCKLES]
    total = sum(weights)
    return Hand(
        ok=True,
        px=px, py=py, conf=list(conf),
        palm=palm,
        ext=tuple(ext),
        pinch_index=gap(INDEX_TIP),
        pinch_middle=gap(MIDDLE_TIP),
        pinch_ring=gap(RING_TIP),
        pinch_little=gap(LITTLE_TIP),
        point_x=(frame.x[THUMB_TIP] + frame.x[INDEX_TIP]) * 0.5,
        point_y=(frame.y[THUMB_TIP] + frame.y[INDEX_TIP]) * 0.5,
        palm_x=sum(px[i] for i in palm_ids) / len(palm_ids),
        palm_y=sum(py[i] for i in palm_ids) / len(palm_ids),
        thumb_dir=(tdx / tlen, tdy / tlen),
        quality=quality,
        knuckle_x=sum(frame.x[i] * w for i, w in zip(KNUCKLES, weights)) / total,
        knuckle_y=sum(frame.y[i] * w for i, w in zip(KNUCKLES, weights)) / total,
    )

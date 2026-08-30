"""Where the screen is, in 3D, and how a hand points at it.

Everything about how the system *feels* runs through this file, so it is worth
being clear about what it does and why it is built this way.

The screen is calibrated as a real rectangle in space: you point at each corner
from two noticeably different hand positions, and each corner is recovered as
the intersection of the two rays. Four corners then give a plane, an origin and
two axes -- a full 3D model of the display.

Doing it this way, rather than mapping hand position straight to screen
position, is what makes several interaction styles possible from one
calibration. A plane in space can be intersected by a ray (pointing), spanned
by a region (absolute reach), or used as a reference for relative motion, and
you can switch between those without recalibrating.

Two rays per corner rather than one because a single ray fixes only a direction.
The intersection of two rays from different positions is what actually locates a
point in space, and it is the same triangulation the cameras themselves rely on.
"""
from __future__ import annotations

import json
import os

import numpy as np

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def closest_point_to_rays(origins: np.ndarray, directions: np.ndarray) -> np.ndarray | None:
    """Least-squares point nearest a bundle of rays.

    Each ray contributes the constraint that the point should lie on it. The
    component of the error along a ray is unconstrained, so each ray is weighted
    by the projector onto its perpendicular plane, and the stacked system is
    solved directly. Two non-parallel rays are enough; more only helps.
    """
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        return None
    directions = directions / norms

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        P = np.eye(3) - np.outer(d, d)      # projector perpendicular to the ray
        A += P
        b += P @ o
    # A is singular when every ray is parallel, which is exactly the case where
    # the rays share a direction and cannot fix a point.
    if np.linalg.cond(A) > 1e8:
        return None
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


class ScreenGeometry:
    """The display as a rectangle in space, plus ray/point mapping onto it."""

    def __init__(self, width_px: int, height_px: int):
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.corners: np.ndarray | None = None      # (4,3) TL, TR, BR, BL
        self.origin: np.ndarray | None = None
        self.axis_u: np.ndarray | None = None       # unit vector along the top edge
        self.axis_v: np.ndarray | None = None       # unit vector down the left edge
        self.normal: np.ndarray | None = None
        self.size_m: tuple[float, float] = (0.0, 0.0)

    @property
    def calibrated(self) -> bool:
        return self.corners is not None

    def set_corners(self, corners: np.ndarray) -> bool:
        """Build the plane and axes from four measured corner positions."""
        corners = np.asarray(corners, dtype=np.float64).reshape(4, 3)
        if not np.all(np.isfinite(corners)):
            return False

        tl, tr, br, bl = corners
        u = ((tr - tl) + (br - bl)) / 2.0     # averaged, since corners are noisy
        v = ((bl - tl) + (br - tr)) / 2.0
        wu, wv = np.linalg.norm(u), np.linalg.norm(v)
        if wu < 1e-4 or wv < 1e-4:
            return False

        axis_u = u / wu
        # Force the vertical axis perpendicular to the horizontal one. The four
        # measured corners never form an exact rectangle, and letting the axes
        # stay skewed makes the mapping shear -- moving your hand horizontally
        # would drift the cursor vertically, which reads as broken tracking.
        axis_v = v - np.dot(v, axis_u) * axis_u
        nv = np.linalg.norm(axis_v)
        if nv < 1e-4:
            return False
        axis_v = axis_v / nv

        self.corners = corners
        self.origin = tl
        self.axis_u, self.axis_v = axis_u, axis_v
        self.normal = np.cross(axis_u, axis_v)
        self.size_m = (float(wu), float(wv))
        return True

    def plane_to_screen(self, point: np.ndarray) -> tuple[float, float] | None:
        """Project a 3D point onto the screen plane and convert to pixels."""
        if not self.calibrated:
            return None
        rel = np.asarray(point, dtype=np.float64) - self.origin
        u = float(np.dot(rel, self.axis_u)) / max(self.size_m[0], 1e-6)
        v = float(np.dot(rel, self.axis_v)) / max(self.size_m[1], 1e-6)
        return self._to_pixels(u, v)

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray,
                      allow_behind: bool = False) -> tuple[float, float] | None:
        """Where a ray meets the screen, in pixels."""
        if not self.calibrated:
            return None
        d = np.asarray(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        d = d / n
        denom = float(np.dot(self.normal, d))
        if abs(denom) < 1e-6:
            return None                      # ray runs parallel to the screen
        o = np.asarray(origin, dtype=np.float64)
        t = float(np.dot(self.normal, self.origin - o)) / denom
        if t <= 0 and not allow_behind:
            return None                      # screen is behind the pointing hand
        return self.plane_to_screen(o + d * t)

    def _to_pixels(self, u: float, v: float) -> tuple[float, float]:
        # Clamped rather than rejected: a cursor pinned to the edge is far more
        # usable than one that vanishes whenever you overshoot slightly.
        x = float(np.clip(u, 0.0, 1.0)) * (self.width_px - 1)
        y = float(np.clip(v, 0.0, 1.0)) * (self.height_px - 1)
        return x, y

    def diagonal_inches(self) -> float:
        """Recovered screen diagonal -- a direct sanity check on calibration."""
        if not self.calibrated:
            return float("nan")
        w, h = self.size_m
        return float(np.hypot(w, h) / 0.0254)

    def to_dict(self) -> dict:
        return {"corners": self.corners.tolist(),
                "screen": [self.width_px, self.height_px]}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str) -> bool:
        try:
            with open(path) as f:
                d = json.load(f)
            return self.set_corners(np.array(d["corners"]))
        except Exception:
            return False


class TouchCollector:
    """Locates the screen corners by touching them, not by pointing at them.

    This replaces ray-based corner calibration, and the reason is arithmetic.
    A pointing ray is built from knuckle to fingertip -- a lever of about 7 cm.
    With the 5 mm of landmark noise the tracker actually has, the ray's
    direction is uncertain by roughly four degrees before the user has even
    aimed. Measured, one degree of error moves the cursor 75 pixels, and
    triangulating two such rays gives corner positions tens of centimetres out.
    In practice it produced a 24 x 10 cm "screen" for a real 34 x 19 cm display.

    Touching removes the lever entirely: the fingertip IS the corner, so corner
    accuracy equals tracking accuracy -- millimetres instead of centimetres.
    It also removes the ambiguity about which corner is which, since the finger
    is physically on it, and it needs one pass rather than two.

    The screen being within arm's reach is what makes this possible; it would
    not work for a wall display or in VR, which is the situation ray-based
    calibration is designed for.
    """

    def __init__(self, samples_per_corner: int = 8):
        self.samples_per_corner = int(samples_per_corner)
        self.points: dict[int, list[np.ndarray]] = {}

    def reset(self) -> None:
        self.points.clear()

    def record(self, corner: int, fingertip: np.ndarray) -> int:
        """Add one fingertip sample for a corner. Returns how many it now has."""
        self.points.setdefault(corner, []).append(np.array(fingertip, dtype=np.float64))
        return len(self.points[corner])

    def has(self, corner: int) -> bool:
        return len(self.points.get(corner, [])) >= 1

    @property
    def complete(self) -> bool:
        return all(self.has(c) for c in range(4))

    def next_target(self) -> int | None:
        for c in range(4):
            if not self.has(c):
                return c
        return None

    def solve(self) -> np.ndarray | None:
        """Median fingertip position per corner.

        A median over several frames rather than a single sample, because the
        detector occasionally throws one landmark badly wrong and a mean would
        carry that straight into the screen model.
        """
        if not self.complete:
            return None
        out = np.zeros((4, 3))
        for c in range(4):
            arr = np.stack(self.points[c])
            if not np.all(np.isfinite(arr)):
                arr = arr[np.all(np.isfinite(arr), axis=1)]
            if len(arr) == 0:
                return None
            out[c] = np.median(arr, axis=0)
        return out

    def spread(self, corner: int) -> float:
        """How much the samples for a corner disagree, in metres."""
        arr = self.points.get(corner)
        if not arr or len(arr) < 2:
            return 0.0
        a = np.stack(arr)
        return float(np.linalg.norm(a - np.median(a, axis=0), axis=1).mean())


class CornerCollector:
    """Gathers pointing rays for each screen corner, from two hand positions.

    The second position is not optional. One ray per corner leaves the distance
    to the corner unknown, so the screen could be any size at any distance along
    those directions -- the same scale ambiguity that made the camera rig
    calibration wander.
    """

    PASSES = 2

    def __init__(self):
        self.samples: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self.samples.clear()

    def record(self, corner: int, pass_index: int,
               origin: np.ndarray, direction: np.ndarray) -> None:
        self.samples[(corner, pass_index)] = (np.array(origin, dtype=np.float64),
                                              np.array(direction, dtype=np.float64))

    def has(self, corner: int, pass_index: int) -> bool:
        return (corner, pass_index) in self.samples

    @property
    def complete(self) -> bool:
        return all(self.has(c, p) for c in range(4) for p in range(self.PASSES))

    def next_target(self) -> tuple[int, int] | None:
        """Which corner and pass to ask for next."""
        for p in range(self.PASSES):
            for c in range(4):
                if not self.has(c, p):
                    return c, p
        return None

    def solve(self) -> np.ndarray | None:
        """Triangulate all four corners. Returns (4,3) or None."""
        if not self.complete:
            return None
        out = np.zeros((4, 3))
        for c in range(4):
            origins = [self.samples[(c, p)][0] for p in range(self.PASSES)]
            dirs = [self.samples[(c, p)][1] for p in range(self.PASSES)]
            point = closest_point_to_rays(np.array(origins), np.array(dirs))
            if point is None:
                return None
            out[c] = point
        return out

    def separation(self, corner: int) -> float:
        """How far apart the two pointing positions were, in metres.

        Reported because it is the quantity that determines whether the corner
        is well located. Two rays from nearly the same place intersect at a
        shallow angle and their crossing point is poorly conditioned -- the same
        reason a stereo rig needs a baseline.
        """
        if not all(self.has(corner, p) for p in range(self.PASSES)):
            return 0.0
        a = self.samples[(corner, 0)][0]
        b = self.samples[(corner, 1)][0]
        return float(np.linalg.norm(a - b))

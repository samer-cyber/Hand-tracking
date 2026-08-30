"""Anatomical constraints on a hand pose.

Temporal smoothing fights jitter by blending across time, and always costs lag.
Geometric constraints fight it for free: a hand whose bones apparently changed
length between two frames is not a fast hand, it is a measurement error, and we
can reject the impossible component of the estimate on the frame it arrives.
In practice this removes the boiling-fingers artifact that dominates the look
of cheap hand tracking, and it removes it without delaying anything.

Two constraints are applied:

1. Bone lengths are fixed. Finger bones do not change length, so we learn their
   lengths for this specific hand and project each frame back onto the set of
   poses that respect them.
2. The palm is rigid. The wrist and the five knuckles move as one body, so we
   fit a rigid transform of a learned palm template rather than trusting five
   independently-noisy points.
"""
from __future__ import annotations

import numpy as np

from .detect import BONES

# Wrist plus the five metacarpophalangeal joints -- the near-rigid palm plate.
PALM_IDS = np.array([0, 1, 5, 9, 13, 17])


def kabsch(P: np.ndarray, Q: np.ndarray, weights: np.ndarray | None = None):
    """Optimal rigid transform mapping P onto Q in the least-squares sense.

    Returns (R, t) such that R @ P.T + t best matches Q.
    """
    if weights is None:
        weights = np.ones(len(P))
    w = (weights / weights.sum())[:, None]
    Pc, Qc = (P * w).sum(0), (Q * w).sum(0)
    P0, Q0 = P - Pc, Q - Qc
    H = (P0 * w).T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])           # guard against a reflection
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


class HandModel:
    """Learns the geometry of one specific hand and enforces it each frame.

    Bone lengths accumulate as a running median over frames the detector was
    confident about -- a median rather than a mean, because occasional gross
    misdetections would otherwise drag the learned skeleton permanently wrong.
    """

    def __init__(self, calibration_frames: int = 90):
        self.calibration_frames = int(calibration_frames)
        self._bone_samples: list[np.ndarray] = []
        self._palm_samples: list[np.ndarray] = []
        self.bone_lengths: np.ndarray | None = None
        self.palm_template: np.ndarray | None = None
        # Once frozen the model stops adapting. This matters for drift
        # detection: a model that keeps re-learning from a drifted rig will
        # quietly absorb the error into the bone lengths and hide the very
        # symptom we rely on to notice the phone was knocked.
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    @property
    def calibrated(self) -> bool:
        return self.bone_lengths is not None

    @property
    def progress(self) -> float:
        return min(1.0, len(self._bone_samples) / self.calibration_frames)

    def observe(self, points: np.ndarray) -> None:
        """Feed a metric 3D pose (21,3) to accumulate this hand dimensions."""
        if self.frozen or points is None or not np.all(np.isfinite(points)):
            return
        lengths = np.array([np.linalg.norm(points[c] - points[p]) for c, p in BONES])
        # A plausible human hand has no bone shorter than 5mm or longer than
        # 12cm; anything outside that is a detection failure, not a hand.
        if np.any(lengths < 0.005) or np.any(lengths > 0.12):
            return

        # Per-bone limits are not enough on their own. A reconstruction whose
        # depth is wrong stretches the whole hand while leaving each individual
        # bone under the cap, so the model quietly learns an 18 cm palm, saves
        # it, and then enforces that wrong size on every later frame -- including
        # frames whose depth was fine. Check the overall dimensions too.
        palm = np.linalg.norm(points[9] - points[0])       # wrist to middle knuckle
        if not (0.055 < palm < 0.135):
            return
        span = np.linalg.norm(points.max(0) - points.min(0))
        if not (0.08 < span < 0.32):
            return
        self._bone_samples.append(lengths)
        # Store the palm centred so templates captured at different hand
        # orientations can be averaged meaningfully.
        palm = points[PALM_IDS] - points[PALM_IDS].mean(0)
        self._palm_samples.append(palm)

        if len(self._bone_samples) > 400:
            self._bone_samples.pop(0)
            self._palm_samples.pop(0)

        if len(self._bone_samples) >= self.calibration_frames:
            self.bone_lengths = np.median(np.stack(self._bone_samples), axis=0)
            self._rebuild_palm_template()

    def _rebuild_palm_template(self) -> None:
        """Average palm samples after aligning them to a common reference."""
        samples = self._palm_samples[-self.calibration_frames:]
        ref = samples[-1]
        aligned = []
        for s in samples:
            R, t = kabsch(s, ref)
            aligned.append((R @ s.T).T + t)
        self.palm_template = np.median(np.stack(aligned), axis=0)

    def enforce_bone_lengths(self, points: np.ndarray, iterations: int = 6,
                             anchor_strength: float = 0.35) -> np.ndarray:
        """Project a pose onto the manifold of poses with these bone lengths.

        Jacobi-style distance-constraint relaxation, the same idea as
        position-based dynamics. Each pass nudges both endpoints of every bone
        toward the correct separation; a few passes converge well.
        """
        if self.bone_lengths is None:
            return points
        p = points.astype(np.float64).copy()
        root = p[0].copy()
        for _ in range(iterations):
            delta = np.zeros_like(p)
            count = np.zeros(len(p))
            for k, (c, par) in enumerate(BONES):
                vec = p[c] - p[par]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    continue
                target = self.bone_lengths[k]
                # Split the correction between endpoints, biased toward moving
                # the distal joint since proximal joints are better constrained
                # by the rest of the chain.
                corr = (dist - target) * (vec / dist)
                delta[c] -= corr * 0.7
                delta[par] += corr * 0.3
                count[c] += 1
                count[par] += 1
            nz = count > 0
            p[nz] += delta[nz] / count[nz, None]
            # Keep the hand where it was measured, rather than letting the whole
            # chain drift while satisfying purely relative constraints.
            p[0] += (root - p[0]) * anchor_strength
        return p

    def enforce_palm_rigidity(self, points: np.ndarray, blend: float = 0.7) -> np.ndarray:
        """Replace palm points with a rigid fit of the learned palm plate."""
        if self.palm_template is None:
            return points
        observed = points[PALM_IDS]
        centre = observed.mean(0)
        R, t = kabsch(self.palm_template, observed - centre)
        fitted = (R @ self.palm_template.T).T + t + centre
        out = points.copy()
        out[PALM_IDS] = blend * fitted + (1.0 - blend) * observed
        return out

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Full constraint pass: rigid palm first, then bone lengths."""
        if not self.calibrated:
            return points
        p = self.enforce_palm_rigidity(points)
        return self.enforce_bone_lengths(p)

    def save(self, path: str) -> None:
        np.savez(path, bone_lengths=self.bone_lengths, palm_template=self.palm_template)

    def load(self, path: str) -> bool:
        try:
            d = np.load(path)
            self.bone_lengths = d["bone_lengths"]
            self.palm_template = d["palm_template"]
            return True
        except Exception:
            return False

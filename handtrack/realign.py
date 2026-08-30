"""Re-solve the phone's pose from live hand observations.

The markerless rig calibration is decent but not exact -- it assumes lens
intrinsics from a field-of-view guess, and takes its metric scale from
MediaPipe's estimate of your hand, which carries several percent of error. The
consequence is that the two cameras genuinely disagree about where your hand is.
Fusion blends that disagreement rather than flipping between the two answers,
which is a large improvement, but it cannot invent the correct answer from two
wrong ones. Some residual offset survives, and it grows with the calibration
error.

This corrects the calibration itself instead of hiding it. The laptop camera,
combined with the measured size of your hand, produces a metric 3D pose on its
own. Pair those 3D points with where the phone *actually saw* the same
landmarks and the phone's pose is the solution to a perspective-n-point problem
-- the same computation used to locate a camera relative to a calibration
target, except the target here is your hand.

The laptop is treated as the reference and only the phone is moved. Correcting
both would leave the pair free to drift together, since nothing else fixes the
world frame.
"""
from __future__ import annotations

import numpy as np

import cv2

from .cameras import MonoPoseSolver
from .fusion import CameraRig


class PhoneRealigner:
    """Collects correspondences, then solves for the phone's pose."""

    def __init__(self, rig_a: CameraRig, rig_b: CameraRig, model,
                 target_frames: int = 90):
        self.rig_a = rig_a
        self.rig_b = rig_b
        self.model = model
        self.target = int(target_frames)
        self.points3d: list[np.ndarray] = []
        self.pixels_a: list[np.ndarray] = []
        self.pixels_b: list[np.ndarray] = []
        self._solver = MonoPoseSolver(rig_a.intrinsics)
        self.active = False
        self.message = ""

    def start(self) -> None:
        self.points3d.clear()
        self.pixels_a.clear()
        self.pixels_b.clear()
        self._solver.reset()
        self.active = True
        self.message = "hold your hand where BOTH cameras see it, and move it around"

    def cancel(self) -> None:
        self.active = False
        self.message = ""

    @property
    def progress(self) -> float:
        return min(1.0, len(self.points3d) / float(self.target))

    def observe(self, px_a: np.ndarray | None, px_b: np.ndarray | None,
                world_a: np.ndarray | None) -> bool:
        """Feed one frame. Returns True when enough data has been gathered."""
        if not self.active or px_a is None or px_b is None:
            return False
        if not self.model.calibrated or world_a is None:
            self.message = "waiting for the hand model to finish learning"
            return False

        # The laptop's own metric reconstruction, which is the reference the
        # phone is being aligned to.
        model_pts = self.model.enforce_bone_lengths(world_a - world_a.mean(0))
        res = self._solver.solve(model_pts, px_a)
        if res is None:
            return False
        cam_pts, _, _ = res
        pts_world = self.rig_a.cam_to_world(cam_pts)
        if not np.all(np.isfinite(pts_world)):
            return False

        self.points3d.append(pts_world)
        self.pixels_a.append(px_a.copy())
        self.pixels_b.append(px_b.copy())
        self.message = f"collecting {len(self.points3d)}/{self.target}"
        return len(self.points3d) >= self.target

    def solve(self) -> tuple[CameraRig, dict] | None:
        """Solve for the phone's pose. Returns the corrected rig and a report."""
        if len(self.points3d) < 20:
            self.message = "not enough paired observations"
            return None

        obj = np.concatenate(self.points3d).astype(np.float64)
        img = np.concatenate(self.pixels_b).astype(np.float64)

        # RANSAC because a single frame where the detector mislabelled the hand
        # would otherwise drag the whole solution.
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
            self.rig_b.intrinsics.K, self.rig_b.intrinsics.dist,
            reprojectionError=6.0, confidence=0.999, iterationsCount=300,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 20:
            self.message = "pose solve failed -- try again with more hand movement"
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        old_c, new_c = self.rig_b.centre, -R.T @ t
        moved = float(np.linalg.norm(new_c - old_c))
        dR = R @ self.rig_b.R.T
        rotated = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))

        # --- fix the scale ---------------------------------------------------
        # Reprojection error is blind to scale: a rig twice as large viewing a
        # hand twice as far away produces identical images. So a pose that
        # minimises reprojection is still free to sit anywhere along that axis,
        # which is why repeated realignments can report baselines of 25 cm,
        # 57 cm and 10 cm while all looking equally good on screen.
        #
        # The hand breaks the tie, because its bones have a known length.
        # Scaling the baseline scales the whole reconstruction about camera A,
        # so the factor that makes the triangulated bones match the measured
        # ones is exactly the factor the rig was wrong by.
        scale, scale_iqr = self._solve_scale(R, t)
        if scale is not None:
            t = t * scale
            new_c = -R.T @ t

        new_rig = CameraRig(self.rig_b.intrinsics, R, t, self.rig_b.name)
        bone_resid = self._bone_residual(new_rig)

        # Residual after the fix, as an honest check that it helped.
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec,
                                    self.rig_b.intrinsics.K, self.rig_b.intrinsics.dist)
        resid = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

        report = {
            "frames": len(self.points3d),
            "inliers": int(len(inliers)),
            "total_points": int(len(obj)),
            "moved_cm": moved * 100.0,
            "rotated_deg": rotated,
            "residual_px": float(np.median(resid[inliers.ravel()])),
            "baseline_cm": float(np.linalg.norm(new_c - self.rig_a.centre) * 100.0),
            "angle_deg": float(np.degrees(np.arccos(np.clip(
                (np.trace(R @ self.rig_a.R.T) - 1) / 2, -1, 1)))),
            "scale_applied": float(scale) if scale is not None else float("nan"),
            "scale_iqr": float(scale_iqr) if scale is not None else float("nan"),
            "bone_residual_mm": bone_resid,
        }
        self.active = False
        self.message = "done"
        return new_rig, report

    def _solve_scale(self, R, t):
        """Factor that makes triangulated bone lengths match the measured hand."""
        from .detect import BONES
        from .fusion import triangulate_hand
        probe = CameraRig(self.rig_b.intrinsics, R, t, "probe")
        ratios = []
        for pa, pb in zip(self.pixels_a, self.pixels_b):
            X = triangulate_hand([self.rig_a, probe], [pa, pb])
            if not np.all(np.isfinite(X)):
                continue
            for k, (c, p) in enumerate(BONES):
                L = np.linalg.norm(X[c] - X[p])
                if L > 1e-9:
                    ratios.append(self.model.bone_lengths[k] / L)
        if len(ratios) < 40:
            return None, float("nan")
        iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
        return float(np.median(ratios)), iqr

    def _bone_residual(self, rig_b) -> float:
        """Mean bone-length error after the fix -- the honest quality figure."""
        from .detect import BONES
        from .fusion import triangulate_hand
        errs = []
        for pa, pb in zip(self.pixels_a[:120], self.pixels_b[:120]):
            X = triangulate_hand([self.rig_a, rig_b], [pa, pb])
            if not np.all(np.isfinite(X)):
                continue
            for k, (c, p) in enumerate(BONES):
                errs.append(abs(np.linalg.norm(X[c] - X[p]) - self.model.bone_lengths[k]))
        return float(np.mean(errs) * 1000.0) if errs else float("nan")

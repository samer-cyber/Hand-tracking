"""The solver that combines every available stream into one 3D hand.

Three streams feed this: the laptop's colour camera, the phone's colour camera,
and the phone's TrueDepth depth map. They are not equally good and they are not
always present, so rather than one fixed formula this picks the best available
method each frame and reports which one it used.

    laptop + phone + depth   triangulate, then correct along the phone's rays
                             using measured depth      <- best
    laptop + phone           triangulate               <- very good
    phone + depth            unproject phone pixels by measured depth
                             (full 3D from one view)   <- good, and the reason
                                                          depth is worth having
    laptop only / phone only solve pose against the calibrated hand skeleton
                             (metric, but depth is inferred not measured)

The third mode is the important one. Pure triangulation gives *nothing* when one
camera loses the hand -- and the laptop camera loses it constantly, because its
field of view is narrow and your hand spends much of its time near the edge of
frame or occluded by the screen bezel. Depth turns that dropout into a merely
slightly-worse estimate instead of a gap. That robustness, not raw precision,
is what the TrueDepth sensor buys.

Everything lands in one world frame -- the laptop camera's -- so consumers
downstream never need to know which mode produced a given frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ekf import HandEKF
from .fusion import CameraRig, DepthFuser, refine_with_skeleton, reprojection_errors, triangulate_hand
from .skeleton import HandModel

MODE_NONE = "none"
MODE_MONO_A = "laptop only"
MODE_MONO_B = "phone only"
MODE_DEPTH_B = "phone + depth"
MODE_STEREO = "stereo"
MODE_STEREO_DEPTH = "stereo + depth"


@dataclass
class SolveResult:
    """One frame's answer, plus enough context to judge how much to trust it."""
    points: np.ndarray | None = None      # (21,3) metres in the world (laptop) frame
    mode: str = MODE_NONE
    timestamp: float = 0.0
    reprojection_px: float = float("nan")
    depth_agreement_mm: float = float("nan")
    bone_residual_mm: float = float("nan")
    sigma_mm: float = float("nan")
    diverged: bool = False
    n_depth_samples: int = 0
    views: tuple[bool, bool] = (False, False)

    @property
    def ok(self) -> bool:
        return self.points is not None and np.all(np.isfinite(self.points))


class HandSolver:
    """Fuses whatever streams are available this frame into one metric hand."""

    def __init__(self, rig_a: CameraRig, rig_b: CameraRig | None,
                 model: HandModel | None = None,
                 depth_weight: float = 0.45,
                 max_reprojection_px: float = 12.0):
        self.rig_a = rig_a
        self.rig_b = rig_b
        self.model = model if model is not None else HandModel()
        self.fuser = DepthFuser(sensor_weight=depth_weight)
        self.max_reprojection_px = float(max_reprojection_px)
        self._mono_solvers: dict[str, object] = {}
        self.ekf = HandEKF()
        # Reprojection error above which the state (or the calibration) is
        # considered broken rather than merely noisy.
        self.diverge_px = 40.0
        self._bad_frames = 0
        # When depth last arrived. The phone's colour and depth streams run at
        # different rates, so roughly every other frame has no depth -- but the
        # state is only milliseconds old, and is far better than what the
        # single-view fallback would produce.
        self._last_depth_time: float | None = None
        # Set when the two views disagree so badly that the calibration, not
        # the tracking, must be at fault. Cleared by a successful realign.
        self.rig_suspect = False

    # -- individual strategies ------------------------------------------------

    def _from_depth(self, rig: CameraRig, pixels: np.ndarray,
                    depth: np.ndarray) -> np.ndarray | None:
        """Back-project landmarks along their viewing rays by measured depth.

        This needs no second camera at all: a pixel fixes the direction and the
        depth sample fixes the distance, which together fix the point. Landmarks
        whose depth is missing are filled in afterwards by the skeleton
        constraints, so a partially-sensed hand still yields a complete pose.
        """
        valid = np.isfinite(depth)
        if valid.sum() < 6:
            return None
        rays_world = rig.undistorted_rays(pixels)          # unit vectors, world frame
        centre = rig.centre

        # The rays are unit-length in world space, but `depth` is measured along
        # the camera's Z axis, so scale by the ray's own Z component to convert
        # a depth into a distance along the ray.
        forward = rig.R[2]                                  # camera Z axis in world
        cos = rays_world @ forward
        cos = np.where(np.abs(cos) < 1e-6, np.nan, cos)
        dist = depth / cos

        pts = np.full((21, 3), np.nan)
        good = valid & np.isfinite(dist)
        pts[good] = centre + rays_world[good] * dist[good, None]
        return pts

    def _mono(self, rig: CameraRig, pixels: np.ndarray, key: str,
              world: np.ndarray | None = None) -> np.ndarray | None:
        """Metric pose from one view, using the calibrated hand as a known object.

        Prefers the detector's own articulated 3D estimate as the model to fit.
        It is scale-ambiguous and noisy along its depth axis, which is exactly
        what PnP resolves, but it does capture how the fingers are *currently*
        bent. Falling back to a static rest-pose template instead costs roughly
        two orders of magnitude in accuracy (70 mm versus under 1 mm), because
        the solver is then trying to explain a curled hand with a flat one.
        """
        from .cameras import MonoPoseSolver
        solver = self._mono_solvers.get(key)
        if solver is None:
            solver = MonoPoseSolver(rig.intrinsics)
            self._mono_solvers[key] = solver

        model_pts = None
        if world is not None and np.all(np.isfinite(world)) and np.ptp(world) > 1e-4:
            model_pts = world - world.mean(0)
            if self.model.calibrated:
                # Rescale the detector's guess to this user's measured hand, so
                # the recovered distance is right and not just self-consistent.
                model_pts = self.model.enforce_bone_lengths(model_pts)
        elif self.model.calibrated:
            model_pts = self._skeleton_template()
        if model_pts is None:
            return None

        res = solver.solve(model_pts, pixels)
        if res is None:
            return None
        cam_pts, _, _ = res
        return rig.cam_to_world(cam_pts)

    def _skeleton_template(self) -> np.ndarray | None:
        """A canonical hand with this user's bone lengths, for PnP to fit.

        Built by walking the kinematic chain outward from the wrist using the
        learned bone lengths and the palm template's directions. It is a rest
        pose, so PnP recovers where the hand is rather than exactly how it is
        articulated -- adequate as a fallback when only one view exists, and
        replaced by real geometry the moment a second view returns.
        """
        if self.model.bone_lengths is None:
            return None
        if getattr(self, "_template_cache", None) is not None:
            return self._template_cache
        from .detect import BONES
        pts = np.zeros((21, 3))
        # Spread the five digits into a fan so the template is not degenerate,
        # which would make the PnP solution ill-conditioned.
        for k, (c, p) in enumerate(BONES):
            finger = 0 if c == 0 else (c - 1) // 4
            ang = np.radians(-40 + finger * 20)
            direction = np.array([np.sin(ang), -np.cos(ang), 0.0])
            pts[c] = pts[p] + direction * self.model.bone_lengths[k]
        pts -= pts.mean(0)
        self._template_cache = pts
        return pts

    # -- the frame-level decision --------------------------------------------

    def solve_fused(self, px_a: np.ndarray | None, px_b: np.ndarray | None,
                    depth_b: np.ndarray | None, timestamp: float,
                    world_a: np.ndarray | None = None,
                    world_b: np.ndarray | None = None) -> SolveResult:
        """Measurement-level fusion into a single persistent state.

        Unlike `solve`, which recomputes a whole pose from whichever sensors are
        present and therefore steps whenever that set changes, this maintains
        one estimate and feeds it evidence. Losing a camera changes how tightly
        the state is constrained, not which estimator is being shown, so the
        hand no longer jumps between two positions as the phone comes and goes.
        """
        res = SolveResult(timestamp=timestamp,
                          views=(px_a is not None, px_b is not None))
        n_depth = int(np.isfinite(depth_b).sum()) if depth_b is not None else 0
        res.n_depth_samples = n_depth
        has_depth = depth_b is not None and n_depth >= 6 and self.rig_b is not None

        ekf = self.ekf
        ekf.predict(timestamp)

        # --- seeding and refresh ---------------------------------------------
        # The geometric estimate is recomputed every frame, not only when the
        # filter is empty, so that landmarks which were unsolvable earlier get
        # picked up as soon as geometry can place them.
        ekf.drop_diverged()
        if self.rig_suspect and px_a is not None:
            px_b, depth_b, has_depth = None, None, False

        seed = None
        if px_a is not None and px_b is not None and self.rig_a and self.rig_b:
            seed = triangulate_hand([self.rig_a, self.rig_b], [px_a, px_b])
        elif px_b is not None and has_depth:
            seed = self._from_depth(self.rig_b, px_b, depth_b)
        elif px_a is not None and self.rig_a is not None:
            seed = self._mono(self.rig_a, px_a, "a", world_a)
        elif px_b is not None and self.rig_b is not None:
            seed = self._mono(self.rig_b, px_b, "b", world_b)

        if seed is not None and np.any(np.isfinite(seed)):
            ekf.seed(seed, timestamp, only_missing=ekf.initialised)
        if not ekf.initialised:
            res.mode = MODE_NONE
            return res

        # --- evidence --------------------------------------------------------
        used = []
        if px_a is not None and self.rig_a is not None:
            if ekf.update_camera(self.rig_a, px_a, sigma_px=2.0):
                used.append("laptop")
        if px_b is not None and self.rig_b is not None:
            if ekf.update_camera(self.rig_b, px_b, sigma_px=2.0):
                used.append("phone")
        if has_depth:
            if ekf.update_depth(self.rig_b, depth_b, sigma_m=0.006):
                used.append("depth")
                self._last_depth_time = timestamp

        # With only one camera and no depth the viewing ray is unconstrained, so
        # a pose solved against the known hand size is needed to pin it.
        #
        # Only when depth is genuinely absent, though. Depth arriving on
        # alternate frames is normal, and on the frames between, the state still
        # carries a depth measurement from ~16 ms ago. Substituting the weaker
        # single-view estimate there would inject its error at half the frame
        # rate, which reads as a buzz at the beat frequency between the streams.
        depth_is_stale = (self._last_depth_time is None
                          or (timestamp - self._last_depth_time) > 0.12)
        if (len(used) == 1 and used[0] in ("laptop", "phone")
                and not has_depth and depth_is_stale):
            rig = self.rig_a if used[0] == "laptop" else self.rig_b
            px = px_a if used[0] == "laptop" else px_b
            world = world_a if used[0] == "laptop" else world_b
            key = "a" if used[0] == "laptop" else "b"
            mono = self._mono(rig, px, key, world)
            if mono is not None and np.all(np.isfinite(mono)):
                ekf.update_position(mono, sigma_m=0.015)
                used.append("anatomy")

        if not ekf.initialised:
            res.mode = MODE_NONE
            return res

        points = ekf.positions()
        if not np.any(np.isfinite(points)):
            res.mode = MODE_NONE
            return res

        # Reprojection is needed below to decide whether the rig is trustworthy
        # enough to learn from, so it is measured before the anatomy step.
        if px_a is not None and px_b is not None and self.rig_a and self.rig_b:
            errs = reprojection_errors([self.rig_a, self.rig_b], [px_a, px_b], points)
            res.reprojection_px = float(np.nanmedian(errs))

        # --- anatomy ---------------------------------------------------------
        # Learn the hand from the detector's own metric estimate, which needs
        # only one camera and no calibration at all.
        #
        # Learning solely from triangulated points, as this used to, creates a
        # deadlock: triangulation needs a good rig, a good rig needs the hand
        # model for its scale, and the mono fallback needs the model too. If the
        # rig starts out bad, the triangulated bones come out implausible, every
        # sample is rejected as anatomically impossible, and calibration sits at
        # 0% forever with no way to recover. Seeding from the single-view
        # estimate breaks that cycle -- it is slightly less precise, but it is
        # always available.
        for w in (world_a, world_b):
            if w is not None and np.all(np.isfinite(w)) and np.ptp(w) > 1e-4:
                self.model.observe(w)
                break

        # Once the rig has proved itself, prefer real triangulated geometry:
        # it measures this hand directly instead of inheriting the detector's
        # learned prior about hands in general.
        if (not self.rig_suspect and "laptop" in used and "phone" in used
                and np.all(np.isfinite(points))
                and np.isfinite(res.reprojection_px) and res.reprojection_px < 6.0):
            self.model.observe(points - np.nanmean(points, axis=0))
        res.bone_residual_mm = self._bone_residual(points)
        if self.model.calibrated and np.all(np.isfinite(points)):
            constrained = refine_with_skeleton(points, self.model.bone_lengths, [], [])
            ekf.apply_constraint(constrained, strength=0.4)
            points = ekf.positions()

        if np.any(~np.isfinite(points)):
            points = self._fill_gaps(points)

        if px_a is not None and px_b is not None and self.rig_a and self.rig_b:
            errs = reprojection_errors([self.rig_a, self.rig_b], [px_a, px_b], points)
            res.reprojection_px = float(np.nanmedian(errs))

            # A reprojection error this large is not noise. Either the state has
            # diverged or, far more often, the extrinsic calibration is simply
            # wrong -- and two cameras that disagree by a hundred pixels cannot
            # be averaged into a sensible answer. Fusing them anyway is what
            # makes the points fly apart.
            #
            # So after a sustained run of bad frames the phone is dropped and
            # the system falls back to the laptop alone, which is less accurate
            # but coherent. That is strictly better than a confident average of
            # two irreconcilable measurements, and it is recoverable: a realign
            # clears the flag.
            if np.isfinite(res.reprojection_px) and res.reprojection_px > self.diverge_px:
                self._bad_frames += 1
                if self._bad_frames >= 15 and not self.rig_suspect:
                    self.rig_suspect = True
                    ekf.reset()
                    res.diverged = True
                    res.mode = MODE_NONE
                    return res
            else:
                self._bad_frames = max(0, self._bad_frames - 1)

        res.mode = "+".join(used) if used else MODE_NONE
        res.points = points
        res.sigma_mm = float(np.nanmean(ekf.position_sigma()) * 1000.0)
        return res

    def _bone_residual(self, points: np.ndarray) -> float:
        """Mean absolute deviation of reconstructed bone lengths, in mm.

        Near zero when the rig is right. Grows steadily as the calibration
        drifts, which makes it the practical way to notice that the phone has
        been knocked before the tracking quietly gets worse.
        """
        if self.model.bone_lengths is None:
            return float("nan")
        from .detect import BONES
        errs = []
        for k, (c, p) in enumerate(BONES):
            if np.all(np.isfinite(points[c])) and np.all(np.isfinite(points[p])):
                errs.append(abs(np.linalg.norm(points[c] - points[p]) - self.model.bone_lengths[k]))
        return float(np.mean(errs) * 1000.0) if errs else float("nan")

    def _fill_gaps(self, points: np.ndarray) -> np.ndarray:
        """Place unmeasured landmarks using bone lengths from their parents."""
        from .detect import BONES, PARENT
        out = points.copy()
        if self.model.bone_lengths is None:
            return out
        lengths = {c: self.model.bone_lengths[k] for k, (c, p) in enumerate(BONES)}
        for _ in range(3):
            for c, p in BONES:
                if np.all(np.isfinite(out[c])) or not np.all(np.isfinite(out[p])):
                    continue
                # No direction information for this joint, so extend along the
                # parent's own bone; crude, but it keeps the skeleton complete
                # and the filter fed until the landmark is seen again.
                gp = int(PARENT[p])
                if gp >= 0 and np.all(np.isfinite(out[gp])):
                    d = out[p] - out[gp]
                    n = np.linalg.norm(d)
                    if n > 1e-6:
                        out[c] = out[p] + d / n * lengths[c]
        return out

"""Measurement-level sensor fusion: one state, many observations.

This replaces the mode-switching estimator, and the reason is the artifact you
can see on screen. The old design computed a *complete* 3D pose from whichever
combination of sensors was available that frame -- stereo, or phone-plus-depth,
or laptop alone -- and then switched between those answers. Each of those
estimators has its own bias, so every switch was a step discontinuity. The hand
appeared to jump between two positions, and the more marginal the phone's view
the more often it happened.

Fusing at the measurement level removes the failure mode rather than smoothing
over it. There is exactly one estimate of where your hand is. Each camera
contributes *evidence* about it -- a 2D observation, or a depth sample -- and
the state absorbs each piece weighted by how much it is trusted. A camera that
loses sight of your hand simply stops contributing evidence; the state carries
on from its motion model with growing uncertainty. Nothing jumps, because
there is no second estimate to jump to.

Each landmark gets an independent constant-velocity Kalman filter with a
6-dimensional state (position and velocity). The filters are independent
because the coupling between landmarks is anatomical rather than dynamic, and
that coupling is better imposed afterwards as a hard constraint on bone lengths
than smeared into a covariance matrix.

The filter is *extended* because the measurement is nonlinear: a camera does
not observe position, it observes a perspective projection of it. Each update
therefore linearises the projection about the current estimate.
"""
from __future__ import annotations

import numpy as np

from .fusion import CameraRig


class HandEKF:
    """21 independent constant-velocity EKFs fused with multi-view evidence."""

    def __init__(
        self,
        process_noise: float = 6.0,        # m/s^2, how hard the hand can accelerate
        initial_pos_var: float = 0.25,     # metres^2, before anything is observed
        initial_vel_var: float = 1.0,
        max_coast_seconds: float = 0.35,
        velocity_tau: float = 0.25,
        max_speed: float = 3.0,
    ):
        self.q = float(process_noise)
        self.initial_pos_var = float(initial_pos_var)
        self.initial_vel_var = float(initial_vel_var)
        self.max_coast = float(max_coast_seconds)
        # Time constant for velocity decay, and a hard physical ceiling. Human
        # fingertips do not exceed a few metres per second, so a state claiming
        # more is diverging, not moving.
        self.velocity_tau = float(velocity_tau)
        self.max_speed = float(max_speed)

        self.x = np.zeros((21, 6))              # [px,py,pz,vx,vy,vz] per landmark
        self.P = np.zeros((21, 6, 6))
        self.valid = np.zeros(21, dtype=bool)
        self.last_t: float | None = None
        self.time_since_update = 0.0

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        self.valid[:] = False
        self.last_t = None
        self.time_since_update = 0.0

    @property
    def initialised(self) -> bool:
        return bool(self.valid.any())

    def positions(self) -> np.ndarray:
        out = np.full((21, 3), np.nan)
        out[self.valid] = self.x[self.valid, :3]
        return out

    def velocities(self) -> np.ndarray:
        out = np.zeros((21, 3))
        out[self.valid] = self.x[self.valid, 3:]
        return out

    def position_sigma(self) -> np.ndarray:
        """Per-landmark positional standard deviation in metres."""
        out = np.full(21, np.nan)
        for i in np.where(self.valid)[0]:
            out[i] = np.sqrt(max(np.trace(self.P[i, :3, :3]) / 3.0, 0.0))
        return out

    def seed(self, points: np.ndarray, t: float, only_missing: bool = False) -> None:
        """Initialise from a 3D estimate (typically triangulation).

        With `only_missing`, landmarks already being tracked are left alone and
        just the untracked ones are brought in. That refresh is not optional:
        `initialised` is true as soon as a *single* landmark is valid, so
        without it any landmark that happened to be unsolvable on the seeding
        frame would stay invalid forever. The output would then always contain
        NaNs and be rejected wholesale -- which shows up as a solve rate of zero
        even while the mode line cheerfully reports both cameras contributing.
        """
        good = np.all(np.isfinite(points), axis=1)
        if only_missing:
            good &= ~self.valid
        for i in np.where(good)[0]:
            self.x[i, :3] = points[i]
            self.x[i, 3:] = 0.0
            self.P[i] = np.diag([self.initial_pos_var] * 3 + [self.initial_vel_var] * 3)
        self.valid |= good
        if self.last_t is None:
            self.last_t = t
        self.time_since_update = 0.0

    def drop_diverged(self, max_sigma_m: float = 0.30) -> int:
        """Invalidate landmarks whose uncertainty has grown implausible.

        A landmark the cameras have effectively stopped constraining will keep
        inflating until it is nowhere near the hand. Better to drop it and let
        it be re-seeded from geometry than to keep drawing it.
        """
        n = 0
        for i in np.where(self.valid)[0]:
            if np.sqrt(max(np.trace(self.P[i, :3, :3]) / 3.0, 0.0)) > max_sigma_m:
                self.valid[i] = False
                n += 1
        return n

    # -- prediction -----------------------------------------------------------

    def predict(self, t: float) -> None:
        if self.last_t is None:
            self.last_t = t
            return
        dt = t - self.last_t
        self.last_t = t
        if dt <= 0:
            return
        dt = float(np.clip(dt, 1e-4, 0.2))
        self.time_since_update += dt

        # Coasting on the motion model is only credible briefly. Beyond that the
        # extrapolation is fiction, and it is better to admit the hand is lost
        # than to keep drawing a confident wrong answer.
        if self.time_since_update > self.max_coast:
            self.valid[:] = False
            return

        # Velocity decays toward zero rather than persisting indefinitely.
        # This matters because a single camera cannot observe depth: its 2D
        # measurements pin two axes and leave the viewing ray free, so with an
        # undamped constant-velocity model the state random-walks along that ray
        # and the velocity estimate grows without bound. Measured, that
        # divergence reached 924 mm within two seconds of the second camera
        # dropping out. Damping bounds the extrapolation while still letting the
        # filter coast usefully through short gaps.
        decay = float(np.exp(-dt / self.velocity_tau))

        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        F[3, 3] = F[4, 4] = F[5, 5] = decay

        # Continuous white-noise acceleration model.
        q = self.q ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = np.zeros((6, 6))
        for k in range(3):
            Q[k, k] = q * dt4 / 4.0
            Q[k, k + 3] = Q[k + 3, k] = q * dt3 / 2.0
            Q[k + 3, k + 3] = q * dt2

        idx = np.where(self.valid)[0]
        if len(idx) == 0:
            return
        self.x[idx] = self.x[idx] @ F.T
        self.P[idx] = F @ self.P[idx] @ F.T + Q

        # Hard ceiling on speed, as a last line of defence against divergence
        # along an unobserved axis.
        speed = np.linalg.norm(self.x[idx, 3:], axis=1)
        hot = speed > self.max_speed
        if np.any(hot):
            rows = idx[hot]
            self.x[rows, 3:] *= (self.max_speed / speed[hot])[:, None]

    # -- measurement updates --------------------------------------------------

    def update_camera(self, rig: CameraRig, pixels: np.ndarray,
                      sigma_px: float = 2.0, gate_px: float = 60.0) -> int:
        """Fold in one camera's 2D observations.

        A landmark is skipped when its predicted projection disagrees with the
        observation by more than `gate_px`. That gate is what stops a single
        mis-detection -- the detector latching onto a face or the other hand --
        from yanking the whole state somewhere wrong. Rejecting an outlier costs
        one frame of information; accepting one costs a visible glitch.
        """
        if pixels is None:
            return 0
        K = rig.intrinsics.K
        fx, fy = K[0, 0], K[1, 1]
        R = rig.R
        Rmeas = np.eye(2) * (sigma_px ** 2)
        used = 0

        for i in np.where(self.valid)[0]:
            p_cam = R @ self.x[i, :3] + rig.t
            z = p_cam[2]
            if z < 0.05:                      # behind or on top of the camera
                continue
            proj = np.array([fx * p_cam[0] / z + K[0, 2],
                             fy * p_cam[1] / z + K[1, 2]])
            innovation = pixels[i] - proj
            if not np.all(np.isfinite(innovation)) or np.linalg.norm(innovation) > gate_px:
                continue

            # d(pixel)/d(camera point), then chain through the world rotation.
            J = np.array([[fx / z, 0.0, -fx * p_cam[0] / (z * z)],
                          [0.0, fy / z, -fy * p_cam[1] / (z * z)]])
            H = np.zeros((2, 6))
            H[:, :3] = J @ R

            self._kalman_update(i, H, innovation, Rmeas)
            used += 1

        if used:
            self.time_since_update = 0.0
        return used

    def update_depth(self, rig: CameraRig, depth: np.ndarray,
                     sigma_m: float = 0.006, gate_m: float = 0.10) -> int:
        """Fold in directly measured depth along a camera's optical axis."""
        if depth is None:
            return 0
        R = rig.R
        Rmeas = np.array([[sigma_m ** 2]])
        used = 0
        for i in np.where(self.valid)[0]:
            d = depth[i]
            if not np.isfinite(d):
                continue
            z_pred = (R @ self.x[i, :3] + rig.t)[2]
            innovation = np.array([d - z_pred])
            if abs(innovation[0]) > gate_m:
                continue
            H = np.zeros((1, 6))
            H[0, :3] = R[2]
            self._kalman_update(i, H, innovation, Rmeas)
            used += 1
        if used:
            self.time_since_update = 0.0
        return used

    def update_position(self, points: np.ndarray, sigma_m: float = 0.015,
                        gate_m: float = 0.25) -> int:
        """Fold in a full 3D position estimate as a measurement.

        This is how depth stays observable when only one camera can see the
        hand. Two-dimensional observations from a single view constrain only
        the two axes across the ray, leaving depth free -- and a filter with an
        unobserved axis does not merely stop improving there, it wanders, since
        process noise keeps inflating that direction with nothing to correct it.

        Solving perspective-n-point against the hand's *known size* resolves it:
        the hand is a calibration object whose dimensions were measured earlier,
        so its apparent size fixes its distance. Feeding that solution in as a
        position measurement -- with a deliberately larger sigma than a stereo
        fix deserves, because it is a weaker kind of evidence -- keeps the state
        anchored without pretending it is as trustworthy as triangulation.

        Because this enters as evidence rather than as a replacement estimate,
        gaining or losing the second camera changes only how sharply the state
        is constrained, never which answer is being displayed. That is what
        removes the visible jump.
        """
        if points is None:
            return 0
        Rm = np.eye(3) * (sigma_m ** 2)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        used = 0
        for i in np.where(self.valid)[0]:
            if not np.all(np.isfinite(points[i])):
                continue
            innovation = points[i] - self.x[i, :3]
            if np.linalg.norm(innovation) > gate_m:
                continue
            self._kalman_update(i, H, innovation, Rm)
            used += 1
        if used:
            self.time_since_update = 0.0
        return used

    def _kalman_update(self, i: int, H: np.ndarray, y: np.ndarray, Rm: np.ndarray) -> None:
        P = self.P[i]
        S = H @ P @ H.T + Rm
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x[i] = self.x[i] + K @ y
        # Joseph form: numerically stable, and keeps the covariance symmetric
        # positive-definite over long runs where the naive form can drift.
        I_KH = np.eye(6) - K @ H
        self.P[i] = I_KH @ P @ I_KH.T + K @ Rm @ K.T

    def spawn_from_rays(self, rig: CameraRig, pixels: np.ndarray,
                        depth: np.ndarray | None, t: float) -> int:
        """Initialise landmarks not yet tracked, when depth makes it possible.

        With a depth sample a single view is enough to place a point, so the
        filter can start without ever having had two cameras agree.
        """
        if pixels is None or depth is None:
            return 0
        rays = rig.undistorted_rays(pixels)
        centre = rig.centre
        forward = rig.R[2]
        n = 0
        for i in range(21):
            if self.valid[i] or not np.isfinite(depth[i]):
                continue
            cos = float(rays[i] @ forward)
            if abs(cos) < 1e-6:
                continue
            self.x[i, :3] = centre + rays[i] * (depth[i] / cos)
            self.x[i, 3:] = 0.0
            self.P[i] = np.diag([self.initial_pos_var] * 3 + [self.initial_vel_var] * 3)
            self.valid[i] = True
            n += 1
        if n:
            self.last_t = t if self.last_t is None else self.last_t
        return n

    def apply_constraint(self, constrained: np.ndarray, strength: float = 0.5,
                         shrink_covariance: float = 0.8) -> None:
        """Push the state toward an anatomically corrected pose.

        This is load-bearing, not cosmetic. With one camera the depth axis is
        unobservable from images alone, but it is *not* unobservable in
        general: a hand drifting along the viewing ray would have to change
        size to keep projecting the same way, and its bones cannot change
        length. Enforcing the learned bone lengths therefore pins the very
        degree of freedom the cameras leave free, which is what stops the
        estimate sliding away when the second view is lost.

        The covariance is shrunk alongside the correction so the filter records
        that it has genuinely gained information, rather than repeatedly
        applying the same correction to a state that still believes it is lost.
        """
        good = np.all(np.isfinite(constrained), axis=1) & self.valid
        for i in np.where(good)[0]:
            self.x[i, :3] += (constrained[i] - self.x[i, :3]) * strength
            self.P[i, :3, :3] *= shrink_covariance

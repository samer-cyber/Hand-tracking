"""Camera intrinsics and monocular metric pose recovery.

A single camera cannot measure absolute depth -- a small hand nearby and a large
hand far away project identically. But once we have learned the bone lengths of
*your* hand (see skeleton.py), that ambiguity is resolved: the hand becomes a
calibration object of known size, and solving the perspective-n-point problem
between the known 3D skeleton and the observed 2D landmarks yields the hand
position in real metres in camera coordinates.

This is why bone-length calibration is load-bearing rather than cosmetic. It is
also why the mono path and the stereo path share a representation: both end up
producing metric 3D points in a camera frame, so everything downstream
(filtering, screen mapping, gestures) is written once.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np


class PinholeCamera:
    """Intrinsic model for one camera."""

    def __init__(self, K: np.ndarray, dist: np.ndarray | None = None,
                 width: int = 0, height: int = 0, name: str = "cam"):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
        self.width = int(width)
        self.height = int(height)
        self.name = name

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_degrees: float = 60.0, name: str = "cam"):
        """Rough intrinsics from an assumed horizontal field of view.

        Good enough to get running; real checkerboard calibration typically
        moves the focal length by a few percent, which matters for absolute
        depth accuracy but not for basic tracking.
        """
        f = (width / 2.0) / np.tan(np.radians(hfov_degrees) / 2.0)
        K = np.array([[f, 0, width / 2.0],
                      [0, f, height / 2.0],
                      [0, 0, 1.0]])
        return cls(K, np.zeros(5), width, height, name)

    @property
    def fx(self) -> float: return float(self.K[0, 0])

    @property
    def fy(self) -> float: return float(self.K[1, 1])

    @property
    def cx(self) -> float: return float(self.K[0, 2])

    @property
    def cy(self) -> float: return float(self.K[1, 2])

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """Project camera-space 3D points to pixels."""
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        px, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    def unproject(self, pts2d: np.ndarray, depth: np.ndarray | float = 1.0) -> np.ndarray:
        """Back-project pixels to camera-space rays scaled by depth (metres)."""
        pts2d = np.asarray(pts2d, dtype=np.float64).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts2d, self.K, self.dist).reshape(-1, 2)
        rays = np.concatenate([undist, np.ones((len(undist), 1))], axis=1)
        d = np.asarray(depth, dtype=np.float64).reshape(-1, 1) if np.ndim(depth) else float(depth)
        return rays * d

    def rescaled(self, width: int, height: int) -> "PinholeCamera":
        """Return these intrinsics adapted to a different frame size.

        Necessary because a phone-webcam app may not deliver the resolution it
        was calibrated at -- Iriun in particular changes it depending on the
        phone-side setting. Intrinsics are in pixels, so a calibration done at
        1280x720 puts the principal point near (640, 360); feed it 1920x1080
        frames and every projection is wrong by hundreds of pixels. That looks
        exactly like a catastrophically bad extrinsic calibration, which is a
        thoroughly misleading place to start debugging.

        Scaling is valid when the change is a resample of the same field of
        view, which is the normal case. A change of FOV (a crop, or a switch
        between lenses) genuinely needs recalibrating.
        """
        if self.width in (0, None) or self.height in (0, None):
            return self
        sx, sy = width / float(self.width), height / float(self.height)
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return PinholeCamera(K, self.dist, width, height, self.name)

    def matches(self, width: int, height: int) -> bool:
        return (self.width, self.height) == (int(width), int(height))

    def to_dict(self) -> dict:
        return {"K": self.K.tolist(), "dist": self.dist.tolist(),
                "width": self.width, "height": self.height, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "PinholeCamera":
        return cls(np.array(d["K"]), np.array(d["dist"]),
                   d.get("width", 0), d.get("height", 0), d.get("name", "cam"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PinholeCamera":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class MonoPoseSolver:
    """Recovers metric hand pose in camera space from 2D landmarks.

    Keeps the previous solution as an initial guess, which both speeds up the
    solve and stops it flipping between the mirror-image poses that perspective
    ambiguity allows when the hand is near-planar to the camera.
    """

    def __init__(self, camera: PinholeCamera):
        self.camera = camera
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

    def reset(self) -> None:
        self._rvec = self._tvec = None

    def solve(self, model_points: np.ndarray, pixels: np.ndarray):
        """Return (points_in_camera_space, rvec, tvec) or None if it fails.

        model_points: (21,3) metric skeleton in a hand-local frame.
        pixels:       (21,2) observed landmark positions.
        """
        obj = np.ascontiguousarray(model_points, dtype=np.float64).reshape(-1, 1, 3)
        img = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if len(obj) < 4:
            return None

        use_guess = self._rvec is not None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, self.camera.K, self.camera.dist,
                rvec=self._rvec.copy() if use_guess else None,
                tvec=self._tvec.copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None
        if not ok:
            self.reset()
            return None

        # solvePnP returns column vectors; flatten before any scalar use, since
        # NumPy 2 refuses to convert a 1-element array to a Python float.
        t = np.asarray(tvec, dtype=np.float64).ravel()

        # A hand behind the camera or 3 metres away is a failed solve, not a
        # pose; reject it rather than letting it poison the temporal filters.
        if not np.all(np.isfinite(t)) or not (0.05 < t[2] < 3.0):
            self.reset()
            return None

        self._rvec, self._tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        cam_points = (R @ model_points.T).T + t
        return cam_points, rvec, tvec

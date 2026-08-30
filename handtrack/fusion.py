"""Multi-view fusion: turning two 2D views plus depth into one metric 3D hand.

This is the mathematical core of the two-camera design, and the reason the rig
is worth building at all.

A single camera measures direction but not distance. Two cameras looking from
different angles each measure a ray through the landmark, and the point where
those rays come closest is the landmark's position in space -- triangulation.
Crucially, the accuracy of that estimate along the depth axis scales with the
angle between the two views, which is exactly why the phone is mounted at an
angle to the laptop rather than beside it. Two cameras pointing the same way
tell you almost nothing extra; two cameras 45 degrees apart turn the worst-
measured axis of one into the best-measured axis of the other.

Three sources of information are combined here, in increasing order of trust:

  1. Ray intersection between the two views (geometric, metric, unbiased).
  2. Direct depth from the phone's TrueDepth sensor where available, which
     measures distance rather than inferring it.
  3. The learned hand skeleton, which rejects the component of any estimate
     that would require bones to change length.

The combination is solved as a weighted least-squares problem rather than by
averaging, so that a landmark seen clearly by one camera and barely by the
other is weighted accordingly instead of being dragged toward the bad estimate.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .cameras import PinholeCamera
from .detect import BONES


class CameraRig:
    """A camera with a known pose in a shared world frame.

    Pose is stored as the transform taking world coordinates into this camera's
    coordinates, which is the convention OpenCV's calibration output uses.
    """

    def __init__(self, intrinsics: PinholeCamera, R: np.ndarray | None = None,
                 t: np.ndarray | None = None, name: str = "cam"):
        self.intrinsics = intrinsics
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).ravel()
        self.name = name

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix mapping world points to homogeneous pixels."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.intrinsics.K @ Rt

    @property
    def centre(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    def world_to_cam(self, pts: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(pts).reshape(-1, 3).T).T + self.t

    def cam_to_world(self, pts: np.ndarray) -> np.ndarray:
        return (self.R.T @ (np.asarray(pts).reshape(-1, 3) - self.t).T).T

    def undistorted_rays(self, pixels: np.ndarray) -> np.ndarray:
        """Unit direction vectors in world space for each pixel."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        norm = cv2.undistortPoints(pts, self.intrinsics.K, self.intrinsics.dist).reshape(-1, 2)
        dirs = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return (self.R.T @ dirs.T).T


def load_rig(path: str) -> tuple[CameraRig, CameraRig, dict]:
    """Load a calibrated two-camera rig produced by tools/calibrate_rig.py.

    Camera A defines the world frame, so it sits at the origin with identity
    rotation and camera B carries the measured relative pose.
    """
    with open(path) as f:
        d = json.load(f)
    a = CameraRig(PinholeCamera.from_dict(d["cam_a"]), np.eye(3), np.zeros(3), "laptop")
    b = CameraRig(PinholeCamera.from_dict(d["cam_b"]),
                  np.array(d["R_b_from_a"]), np.array(d["T_b_from_a"]), "phone")
    return a, b, d


def triangulate_dlt(observations: list[tuple[CameraRig, np.ndarray, float]]) -> np.ndarray:
    """Triangulate one 3D point from N weighted 2D observations.

    Each observation is (rig, pixel, weight). Builds the standard homogeneous
    linear system from the projection matrices and solves it by SVD; the
    solution is the direction of least residual, i.e. the point closest to all
    the viewing rays simultaneously.
    """
    rows = []
    for rig, px, w in observations:
        if w <= 0:
            continue
        P = rig.P
        x, y = float(px[0]), float(px[1])
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
    if len(rows) < 4:
        return np.full(3, np.nan)
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_hand(rigs: list[CameraRig], pixel_sets: list[np.ndarray],
                     weights: list[float] | None = None) -> np.ndarray:
    """Triangulate all 21 landmarks. Returns (21,3) world points, NaN where unsolvable."""
    n = len(rigs)
    if weights is None:
        weights = [1.0] * n
    out = np.full((21, 3), np.nan)
    for j in range(21):
        obs = [(rigs[i], pixel_sets[i][j], weights[i])
               for i in range(n) if pixel_sets[i] is not None]
        if len(obs) >= 2:
            out[j] = triangulate_dlt(obs)
    return out


def reprojection_errors(rigs: list[CameraRig], pixel_sets: list[np.ndarray],
                        points3d: np.ndarray) -> np.ndarray:
    """Per-landmark RMS reprojection error in pixels, across all views.

    This is the honest self-check on a triangulated result: if the two cameras
    disagree about where a landmark is, the error shows up here. Large values
    mean the extrinsic calibration has drifted or one view has mis-detected,
    and the estimate should be distrusted rather than displayed confidently.
    """
    errs = np.full((len(rigs), 21), np.nan)
    for i, rig in enumerate(rigs):
        if pixel_sets[i] is None:
            continue
        valid = np.all(np.isfinite(points3d), axis=1)
        if not valid.any():
            continue
        cam_pts = rig.world_to_cam(points3d[valid])
        proj = rig.intrinsics.project(cam_pts)
        errs[i, valid] = np.linalg.norm(proj - pixel_sets[i][valid], axis=1)
    return np.sqrt(np.nanmean(errs ** 2, axis=0))


class DepthFuser:
    """Blends triangulated depth with directly measured sensor depth.

    Triangulation and the depth sensor fail in different ways, which is what
    makes combining them worthwhile. Triangulation degrades when the landmark
    is near the plane containing both cameras, or when one view mis-detects.
    Structured-light depth degrades at glancing angles, on the edges of the
    hand where the dot pattern falls off, and beyond about a metre. Where both
    are healthy they should agree; where they disagree by a lot, one of them is
    wrong and we trust neither too far.
    """

    def __init__(self, max_disagreement_m: float = 0.05, sensor_weight: float = 0.6):
        self.max_disagreement = float(max_disagreement_m)
        self.sensor_weight = float(sensor_weight)
        self.last_agreement: float = float("nan")

    def fuse(self, triangulated_world: np.ndarray, depth_rig: CameraRig,
             measured_depth: np.ndarray) -> np.ndarray:
        """Pull triangulated points toward the sensor-measured depth.

        `measured_depth` is (21,) metres along the depth camera's Z axis, NaN
        where the sensor returned nothing. Points are moved along the depth
        camera's viewing ray only, so the well-constrained lateral position
        from triangulation is preserved.
        """
        out = triangulated_world.copy()
        cam_pts = depth_rig.world_to_cam(triangulated_world)
        valid = np.isfinite(measured_depth) & np.all(np.isfinite(cam_pts), axis=1)
        valid &= (measured_depth > 0.05) & (measured_depth < 2.0)
        if not valid.any():
            self.last_agreement = float("nan")
            return out

        disagreement = np.abs(cam_pts[valid, 2] - measured_depth[valid])
        self.last_agreement = float(np.median(disagreement))

        # Down-weight individual landmarks whose two estimates are far apart;
        # a large disagreement means one measurement is simply wrong.
        w = self.sensor_weight * np.clip(1.0 - disagreement / self.max_disagreement, 0.0, 1.0)
        scale = np.ones(len(cam_pts))
        z = cam_pts[valid, 2]
        safe = np.abs(z) > 1e-6
        idx = np.where(valid)[0][safe]
        blended_z = (1.0 - w[safe]) * z[safe] + w[safe] * measured_depth[idx]
        scale[idx] = blended_z / z[safe]

        corrected_cam = cam_pts * scale[:, None]
        return depth_rig.cam_to_world(corrected_cam)


def refine_with_skeleton(points: np.ndarray, bone_lengths: np.ndarray | None,
                         rigs: list[CameraRig], pixel_sets: list[np.ndarray],
                         iterations: int = 8, bone_weight: float = 0.5) -> np.ndarray:
    """Jointly satisfy the image observations and the bone-length constraints.

    Alternates between pulling points toward their triangulated positions and
    projecting onto correct bone lengths. This is a cheap alternating solver
    rather than a full bundle adjustment; it converges fast enough to run every
    frame, which matters far more here than squeezing out the last fraction of
    a millimetre.
    """
    if bone_lengths is None:
        return points
    p = points.copy()
    anchor = points.copy()
    valid_anchor = np.all(np.isfinite(anchor), axis=1)

    for _ in range(iterations):
        delta = np.zeros_like(p)
        count = np.zeros(len(p))
        for k, (c, par) in enumerate(BONES):
            if not (np.all(np.isfinite(p[c])) and np.all(np.isfinite(p[par]))):
                continue
            vec = p[c] - p[par]
            dist = np.linalg.norm(vec)
            if dist < 1e-9:
                continue
            corr = (dist - bone_lengths[k]) * (vec / dist) * bone_weight
            delta[c] -= corr * 0.5
            delta[par] += corr * 0.5
            count[c] += 1
            count[par] += 1
        nz = count > 0
        p[nz] += delta[nz] / count[nz, None]
        # Pull back toward the measured positions so the constraints shape the
        # pose without letting it float away from what the cameras actually saw.
        p[valid_anchor] += (anchor[valid_anchor] - p[valid_anchor]) * 0.25
    return p


def adapt_rig_to_sources(rig_a, rig_b, src_a, src_b):
    """Rescale calibrated intrinsics to whatever resolution is actually arriving.

    A silent resolution change between calibration and runtime is one of the few
    faults that produces reprojection errors in the hundreds of pixels, and it is
    invisible unless explicitly checked.
    """
    for rig, src in ((rig_a, src_a), (rig_b, src_b)):
        if rig is None or src is None:
            continue
        w, h = int(src.width), int(src.height)
        intr = rig.intrinsics
        if intr.width and not intr.matches(w, h):
            print(f"  !! {rig.name} was calibrated at {intr.width}x{intr.height} "
                  f"but is streaming {w}x{h}")
            print("     rescaling its intrinsics to match "
                  "(recalibrate if the field of view also changed)")
            rig.intrinsics = intr.rescaled(w, h)
    return rig_a, rig_b

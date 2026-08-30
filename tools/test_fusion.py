"""Synthetic validation of the fusion geometry.

Builds a virtual two-camera rig matching the intended physical setup, projects
a known hand into both views, adds realistic pixel noise, and checks that
triangulation recovers the original 3D points. This is how we know the geometry
is right before any of it meets a real camera, and it also quantifies the payoff
from the 45-degree mounting angle.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import BONES
from handtrack.fusion import CameraRig, DepthFuser, refine_with_skeleton, reprojection_errors, triangulate_hand


def rig_at_angle(angle_deg: float, distance: float, width=1280, height=720, hfov=60.0, name="cam"):
    """A camera positioned `distance` from the origin, rotated by `angle_deg`
    about the vertical axis, and aimed back at the origin."""
    a = np.radians(angle_deg)
    centre = np.array([np.sin(a) * distance, 0.0, -np.cos(a) * distance])
    forward = -centre / np.linalg.norm(centre)
    up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.stack([right, up, forward])       # world -> camera
    t = -R @ centre
    return CameraRig(PinholeCamera.from_fov(width, height, hfov, name), R, t, name)


def make_hand(centre=np.array([0.0, 0.0, 0.0])):
    """A hand with real proportions: about a 9 cm palm and an 18 cm span.

    Sized deliberately. The anatomical gates in HandModel reject implausible
    geometry, so a toy skeleton built from arbitrary offsets is refused -- and a
    test hand that the production code would throw away is testing nothing.
    """
    pts = np.zeros((21, 3))
    spread = 0.085                      # knuckle to knuckle across the palm
    palm = 0.090                        # wrist to middle knuckle
    lengths = [(0.032, 0.028, 0.024),   # thumb
               (0.040, 0.026, 0.020),   # index
               (0.044, 0.028, 0.021),   # middle
               (0.040, 0.026, 0.020),   # ring
               (0.032, 0.021, 0.018)]   # little
    for f in range(5):
        x = -spread / 2 + spread * f / 4
        base = np.array([x, palm * (0.72 if f == 0 else 1.0), 0.004 * f])
        pts[1 + f * 4] = base
        run = base
        for j, seg in enumerate(lengths[f]):
            run = run + np.array([x * 0.18, seg, 0.002])
            pts[2 + f * 4 + j] = run
    return pts - pts.mean(0) + centre


def run(angle: float, noise_px: float, trials: int = 200, use_depth: bool = False):
    rng = np.random.default_rng(0)
    laptop = rig_at_angle(0.0, 0.5, name="laptop")
    phone = rig_at_angle(angle, 0.5, name="phone")
    rigs = [laptop, phone]
    truth = make_hand(np.array([0.0, 0.0, 0.0]))
    bone_lengths = np.array([np.linalg.norm(truth[c] - truth[p]) for c, p in BONES])

    errs, depth_errs, lateral_errs = [], [], []
    for _ in range(trials):
        pixel_sets = []
        for rig in rigs:
            cam_pts = rig.world_to_cam(truth)
            px = rig.intrinsics.project(cam_pts)
            pixel_sets.append(px + rng.normal(0, noise_px, px.shape))

        est = triangulate_hand(rigs, pixel_sets)
        if use_depth:
            measured = phone.world_to_cam(truth)[:, 2] + rng.normal(0, 0.003, 21)
            est = DepthFuser().fuse(est, phone, measured)
        est = refine_with_skeleton(est, bone_lengths, rigs, pixel_sets)

        d = est - truth
        errs.append(np.linalg.norm(d, axis=1))
        depth_errs.append(np.abs(d[:, 2]))
        lateral_errs.append(np.linalg.norm(d[:, :2], axis=1))

    return (np.mean(errs) * 1000, np.mean(depth_errs) * 1000, np.mean(lateral_errs) * 1000)


def main():
    print("Synthetic rig: hand at 50 cm, 1280x720, 60 deg FOV, 1.5 px detector noise\n")
    print(f"{'phone angle':>12} {'3D err':>9} {'depth err':>10} {'lateral err':>12}")
    print("-" * 48)
    for angle in [5, 15, 30, 45, 60, 75]:
        e, d, l = run(angle, noise_px=1.5)
        print(f"{angle:>10} deg {e:>7.2f}mm {d:>8.2f}mm {l:>10.2f}mm")

    print("\nEffect of adding TrueDepth measurements (3 mm sensor noise), at 45 deg:")
    e0, d0, _ = run(45, 1.5, use_depth=False)
    e1, d1, _ = run(45, 1.5, use_depth=True)
    print(f"  triangulation only      3D {e0:6.2f} mm   depth {d0:6.2f} mm")
    print(f"  triangulation + depth   3D {e1:6.2f} mm   depth {d1:6.2f} mm")

    print("\nDetector noise sensitivity at 45 deg (the dominant error source):")
    for npx in [0.5, 1.0, 2.0, 4.0]:
        e, d, l = run(45, npx)
        print(f"  {npx:>4.1f} px noise -> 3D {e:6.2f} mm   depth {d:6.2f} mm")


if __name__ == "__main__":
    main()

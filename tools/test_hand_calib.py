"""Verify the markerless (hand-based) rig calibration against a known rig.

The estimation chain here is easy to get subtly wrong -- essential matrix,
pose recovery, and a scale factor derived from bone lengths, each with its own
sign and frame conventions. So it is checked against a synthetic rig whose
answer we already know, at realistic landmark noise.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import BONES
from handtrack.fusion import CameraRig, triangulate_hand
from test_fusion import make_hand, rig_at_angle


def estimate(pts_a, pts_b, intr_a, intr_b, bone_lengths):
    """Exactly the estimation the real tool performs."""
    A = np.concatenate(pts_a).astype(np.float64)
    B = np.concatenate(pts_b).astype(np.float64)
    na = cv2.undistortPoints(A.reshape(-1, 1, 2), intr_a.K, intr_a.dist).reshape(-1, 2)
    nb = cv2.undistortPoints(B.reshape(-1, 1, 2), intr_b.K, intr_b.dist).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC,
                                   prob=0.9999, threshold=0.003)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
    t = t.ravel() / max(np.linalg.norm(t.ravel()), 1e-12)

    rig_a = CameraRig(intr_a, np.eye(3), np.zeros(3))
    rig_unit = CameraRig(intr_b, R, t)
    ratios = []
    for pa, pb in zip(pts_a, pts_b):
        X = triangulate_hand([rig_a, rig_unit], [pa, pb])
        if not np.all(np.isfinite(X)):
            continue
        for k, (c, p) in enumerate(BONES):
            L = np.linalg.norm(X[c] - X[p])
            if L > 1e-9:
                ratios.append(bone_lengths[k] / L)
    if len(ratios) < 20:
        return None
    return R, t * float(np.median(ratios)), int(mask.sum())


def main():
    rng = np.random.default_rng(0)
    laptop = rig_at_angle(0.0, 0.5, name="laptop")
    phone = rig_at_angle(45.0, 0.5, name="phone")
    truth_R = phone.R @ laptop.R.T
    truth_t = phone.t - truth_R @ laptop.t
    truth_baseline = np.linalg.norm(truth_t)

    hand = make_hand()
    bone_lengths = np.array([np.linalg.norm(hand[c] - hand[p]) for c, p in BONES])

    print(f"Ground truth: baseline {truth_baseline*100:.1f} cm, angle 45.0 deg")
    print("Simulating a hand moved around the shared view.\n")
    print(f"{'noise':>7} {'frames':>7} {'baseline':>10} {'angle err':>10} {'t dir err':>10}")
    print("-" * 50)

    for noise in (0.5, 1.5, 3.0):
        for n_frames in (60, 240):
            pts_a, pts_b = [], []
            for _ in range(n_frames):
                # Move the hand through the shared volume, and rotate it, so the
                # correspondences are not confined to one spot or one pose.
                offset = np.array([rng.uniform(-0.14, 0.14),
                                   rng.uniform(-0.11, 0.11),
                                   rng.uniform(-0.09, 0.09)])
                ang = rng.uniform(-0.9, 0.9)
                Rz = np.array([[np.cos(ang), -np.sin(ang), 0],
                               [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
                H = (Rz @ hand.T).T + offset
                pa = laptop.intrinsics.project(laptop.world_to_cam(H))
                pb = phone.intrinsics.project(phone.world_to_cam(H))
                pts_a.append(pa + rng.normal(0, noise, pa.shape))
                pts_b.append(pb + rng.normal(0, noise, pb.shape))

            res = estimate(pts_a, pts_b, laptop.intrinsics, phone.intrinsics, bone_lengths)
            if res is None:
                print(f"{noise:>6.1f}px {n_frames:>7} {'FAILED':>10}")
                continue
            R, t, inl = res
            dR = R @ truth_R.T
            ang_err = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            dir_err = np.degrees(np.arccos(np.clip(
                np.dot(t / np.linalg.norm(t), truth_t / truth_baseline), -1, 1)))
            print(f"{noise:>6.1f}px {n_frames:>7} {np.linalg.norm(t)*100:>9.1f}cm "
                  f"{ang_err:>9.2f}d {dir_err:>9.2f}d")

    print("\nbaseline should read ~38.3 cm; angle and direction errors should be small.")


if __name__ == "__main__":
    main()

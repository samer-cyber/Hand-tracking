"""Check that phone packets reconstruct the hand that produced them.

Projects a known 3D hand through known intrinsics into landmark pixels plus
per-landmark depth -- exactly what the phone sends -- then runs the same
back-projection, filtering and constraint path the live system uses, and
compares against the truth it started from.

Also checks the case that occurs on roughly half of all real frames: depth
missing, only 2D available. Depth is unobservable from a single view, so the
question is whether the filter coasts sensibly or slides along the ray.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import BONES
from handtrack.ekf import HandEKF
from handtrack.filters import OneEuroFilter
from handtrack.fusion import CameraRig, refine_with_skeleton
from handtrack.skeleton import HandModel
from run_hand import back_project

W, H, FX = 1512, 2016, 1400.0


def project(p3: np.ndarray) -> np.ndarray:
    """Standard pinhole projection, image origin top-left with y increasing down.

    This is the convention the phone sends in: Vision reports normalised points
    with the origin bottom-left, and CaptureEngine converts them with
    (1 - y) * height before transmitting. Getting this flip wrong is silent --
    the hand reconstructs to a plausible-looking but mirrored position.
    """
    return np.stack([p3[:, 0] / p3[:, 2] * FX + W / 2,
                     p3[:, 1] / p3[:, 2] * FX + H / 2], axis=1)


def make_hand(centre):
    rng = np.random.default_rng(5)
    pts = np.zeros((21, 3))
    for c, p in BONES:
        pts[c] = pts[p] + rng.normal(0, 1, 3) * 0.011 + np.array([0.016, 0.014, 0.003])
    return pts - pts.mean(0) + centre


def main():
    rng = np.random.default_rng(0)
    intr = PinholeCamera(np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]]),
                         np.zeros(5), W, H, "phone")
    rig = CameraRig(intr, np.eye(3), np.zeros(3), "phone")

    print(f"Phone {W}x{H}, fx={FX:.0f}, hand at 40 cm\n")

    # --- back-projection alone, no noise ---------------------------------
    truth = make_hand(np.array([0.02, -0.03, 0.40]))
    exact = back_project(rig, project(truth), truth[:, 2])
    print(f"back-projection (exact): max error "
          f"{np.nanmax(np.linalg.norm(exact - truth, axis=1)) * 1000:.4f} mm")

    # --- full path, with the noise the phone actually delivers ------------
    print(f"\n{'depth every':>14} {'3D error':>10} {'bone error':>12}")
    print("-" * 40)

    for label, depth_every in (("frame", 1), ("2nd frame", 2), ("never", 0)):
        model = HandModel(calibration_frames=40)
        ekf = HandEKF()
        filt = OneEuroFilter(1.2, 0.05)
        errs, bone_errs = [], []

        for k in range(220):
            t = k / 60.0
            centre = np.array([0.02 + 0.04 * np.sin(t * 1.3),
                               -0.03 + 0.03 * np.cos(t), 0.40 + 0.02 * np.sin(t * 0.7)])
            truth = make_hand(centre)
            px = project(truth) + rng.normal(0, 1.5, (21, 2))
            has_depth = depth_every and (k % depth_every == 0)
            depth = truth[:, 2] + rng.normal(0, 0.004, 21) if has_depth else None

            ekf.predict(t)
            if depth is not None:
                ekf.seed(back_project(rig, px, depth), t, only_missing=ekf.initialised)
            elif not ekf.initialised:
                continue
            ekf.update_camera(rig, px, sigma_px=2.5)
            if depth is not None:
                ekf.update_depth(rig, depth, sigma_m=0.006)

            raw = ekf.positions()
            if not np.all(np.isfinite(raw)):
                continue
            model.observe(raw - raw.mean(0))
            if model.calibrated:
                ekf.apply_constraint(refine_with_skeleton(raw, model.bone_lengths, [], []),
                                     strength=0.35)
            out = filt(ekf.positions(), t)

            if k > 60:                       # let it converge first
                errs.append(np.linalg.norm(out - truth, axis=1).mean())
                bone_errs.append(np.mean([
                    abs(np.linalg.norm(out[c] - out[p]) - np.linalg.norm(truth[c] - truth[p]))
                    for c, p in BONES]))

        if errs:
            print(f"{label:>14} {np.mean(errs)*1000:7.2f} mm {np.mean(bone_errs)*1000:9.2f} mm")
        else:
            print(f"{label:>14} {'no solution':>21}")

    print("\n'never' is the honest control: with 2D only and no depth ever, the")
    print("distance to the hand is unobservable and the estimate should be poor.")
    print("'2nd frame' is what the phone actually delivers.")

    # --- how far it can coast when depth stops ----------------------------
    print("\nDepth stops after 2 seconds; error over the following frames:")
    model = HandModel(calibration_frames=40)
    ekf, filt = HandEKF(), OneEuroFilter(1.2, 0.05)
    stop_at = 120
    marks = {}
    for k in range(stop_at + 40):
        t = k / 60.0
        truth = make_hand(np.array([0.02, -0.03, 0.40]))
        px = project(truth) + rng.normal(0, 1.5, (21, 2))
        depth = (truth[:, 2] + rng.normal(0, 0.004, 21)) if k < stop_at else None
        ekf.predict(t)
        if depth is not None:
            ekf.seed(back_project(rig, px, depth), t, only_missing=ekf.initialised)
        if not ekf.initialised:
            continue
        ekf.update_camera(rig, px, sigma_px=2.5)
        if depth is not None:
            ekf.update_depth(rig, depth, sigma_m=0.006)
        raw = ekf.positions()
        if not np.all(np.isfinite(raw)):
            continue
        model.observe(raw - raw.mean(0))
        if model.calibrated:
            ekf.apply_constraint(refine_with_skeleton(raw, model.bone_lengths, [], []), 0.35)
        out = filt(ekf.positions(), t)
        after = k - stop_at
        if after in (0, 5, 10, 20, 39):
            marks[after] = np.linalg.norm(out - truth, axis=1).mean() * 1000
    for after, err in sorted(marks.items()):
        print(f"  {after*1000//60:4d} ms after last depth: {err:7.2f} mm")


if __name__ == "__main__":
    main()

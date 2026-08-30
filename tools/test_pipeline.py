"""Validate every solver mode against synthetic ground truth.

Each mode uses different mathematics, so each can be wrong independently. This
checks all of them on a virtual rig matching the intended physical setup, with
realistic noise, and confirms the solver both picks the right mode and gets the
right answer in it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.detect import BONES
from handtrack.pipeline import (MODE_DEPTH_B, MODE_MONO_A, MODE_STEREO,
                                MODE_STEREO_DEPTH, HandSolver)
from handtrack.skeleton import HandModel
from test_fusion import make_hand, rig_at_angle


def build(hand_calibrated: bool = True):
    laptop = rig_at_angle(0.0, 0.5, name="laptop")
    phone = rig_at_angle(45.0, 0.5, name="phone")
    truth = make_hand()
    model = HandModel(calibration_frames=30)
    if hand_calibrated:
        rng = np.random.default_rng(1)
        for _ in range(40):
            model.observe(truth - truth.mean(0) + rng.normal(0, 0.0003, (21, 3)))
    return laptop, phone, truth, model


def project(rig, truth, noise, rng):
    return rig.intrinsics.project(rig.world_to_cam(truth)) + rng.normal(0, noise, (21, 2))


def main():
    rng = np.random.default_rng(0)
    laptop, phone, truth, model = build()
    solver = HandSolver(laptop, phone, model)

    print("Virtual rig: laptop at 0 deg, phone at 45 deg, hand at 50 cm")
    print("Detector noise 1.5 px, depth noise 3 mm\n")
    print(f"{'available streams':<28} {'mode chosen':<18} {'3D error':>10}")
    print("-" * 60)

    def depth_of(truth, noise=0.003):
        return phone.world_to_cam(truth)[:, 2] + rng.normal(0, noise, 21)

    trials = 60
    for label, use_a, use_b, use_d in [
        ("laptop + phone + depth", True, True, True),
        ("laptop + phone", True, True, False),
        ("phone + depth (no laptop)", False, True, True),
        ("laptop only", True, False, False),
    ]:
        errs, modes = [], set()
        for _ in range(trials):
            pa = project(laptop, truth, 1.5, rng) if use_a else None
            pb = project(phone, truth, 1.5, rng) if use_b else None
            dd = depth_of(truth) if use_d else None
            # Stand in for MediaPipe's articulated world landmarks: correct
            # shape, hand-centred, with the noise such an estimate really has.
            wa = (truth - truth.mean(0)) + rng.normal(0, 0.004, (21, 3))
            r = solver.solve_fused(pa, pb, dd, 0.0, world_a=wa, world_b=wa)
            modes.add(r.mode)
            if r.ok:
                errs.append(np.linalg.norm(r.points - truth, axis=1).mean())
        mode = "/".join(sorted(modes))
        e = f"{np.mean(errs)*1000:8.2f} mm" if errs else "   FAILED"
        print(f"{label:<28} {mode:<18} {e:>10}")

    # The mode that exists purely to survive the laptop losing the hand.
    print("\nDepth-only accuracy vs depth sensor noise (phone view only):")
    for dn in [0.001, 0.003, 0.006, 0.012]:
        errs = []
        for _ in range(trials):
            pb = project(phone, truth, 1.5, rng)
            r = solver.solve_fused(None, pb, depth_of(truth, dn), 0.0)
            if r.ok:
                errs.append(np.linalg.norm(r.points - truth, axis=1).mean())
        print(f"  {dn*1000:5.1f} mm sensor noise -> {np.mean(errs)*1000:6.2f} mm 3D error")

    # A drifted calibration must be caught, not silently believed. The realistic
    # failure is the phone being knocked so its AIM changes while it stays put --
    # not orbiting around the hand, which barely disturbs the geometry at all.
    print("\nCalibration-drift detection (phone bumped, aim shifted):")
    print("  observations come from the true rig; the solver believes the drifted one")
    from handtrack.fusion import CameraRig
    for drift in [0.0, 0.5, 1.0, 3.0, 8.0]:
        a = np.radians(drift)
        # Rotate about the camera's own vertical axis: a pure aiming error.
        Rd = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        bad_phone = CameraRig(phone.intrinsics, Rd @ phone.R, phone.t, "phone")
        _, _, _, frozen_model = build()
        frozen_model.freeze()          # so it cannot absorb the drift and hide it
        s2 = HandSolver(laptop, bad_phone, frozen_model)
        reps, errs, bones, depth_dis = [], [], [], []
        for _ in range(trials):
            pa = project(laptop, truth, 1.5, rng)
            pb = project(phone, truth, 1.5, rng)      # true observations, wrong rig
            r = s2.solve_fused(pa, pb, depth_of(truth), 0.0)
            if r.ok:
                if np.isfinite(r.reprojection_px):
                    reps.append(r.reprojection_px)
                if np.isfinite(r.bone_residual_mm):
                    bones.append(r.bone_residual_mm)
                if np.isfinite(r.depth_agreement_mm):
                    depth_dis.append(r.depth_agreement_mm)
                errs.append(np.linalg.norm(r.points - truth, axis=1).mean())
        f = lambda v: np.mean(v) if v else float("nan")
        print(f"  {drift:4.1f} deg | true err {f(errs)*1000:6.2f} mm | "
              f"reproj {f(reps):5.2f} px | bone resid {f(bones):6.2f} mm | "
              f"depth disagree {f(depth_dis):6.2f} mm")


if __name__ == "__main__":
    main()

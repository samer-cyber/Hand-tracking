"""Does measurement-level fusion actually remove the mode-switching jump?

The reported symptom is the hand appearing at two different positions as the
phone drifts in and out of view. This reproduces that: a hand moving on a known
path, with the phone's view dropping out intermittently, and measures the size
of the discontinuity at every transition.

Two conditions are compared. Estimate-level switching recomputes a complete
pose from whatever is available and switches between the answers. Measurement-
level fusion keeps one state and feeds it whatever evidence arrives.

The second scenario is the important one: the same test with a deliberately
miscalibrated rig, since a real hand-based calibration carries several percent
of scale error, and that is precisely when the two estimators disagree most.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import MonoPoseSolver
from handtrack.ekf import HandEKF
from handtrack.fusion import CameraRig, refine_with_skeleton, triangulate_hand
from handtrack.detect import BONES
from test_fusion import make_hand, rig_at_angle


def trajectory(n: int, hand: np.ndarray):
    """A hand sweeping smoothly through the shared volume."""
    for k in range(n):
        t = k / 30.0
        offset = np.array([0.10 * np.sin(t * 1.6),
                           0.06 * np.sin(t * 1.1 + 1.0),
                           0.05 * np.sin(t * 0.7)])
        yield t, hand + offset


def visibility(n: int, seed: int = 3):
    """Phone visibility that flickers, as it does when badly positioned."""
    rng = np.random.default_rng(seed)
    vis, state = [], True
    for _ in range(n):
        if rng.random() < (0.06 if state else 0.18):
            state = not state
        vis.append(state)
    return vis


def run(scale_error: float = 1.0, noise: float = 1.5, n: int = 400):
    rng = np.random.default_rng(0)
    laptop = rig_at_angle(0.0, 0.5, name="laptop")
    phone_true = rig_at_angle(45.0, 0.5, name="phone")
    # The rig the solver believes in, with a scale error on the baseline.
    phone_belief = CameraRig(phone_true.intrinsics, phone_true.R,
                             phone_true.t * scale_error, "phone")

    hand = make_hand()
    bones = np.array([np.linalg.norm(hand[c] - hand[p]) for c, p in BONES])
    vis = visibility(n)

    ekf = HandEKF()
    solver_mono = MonoPoseSolver(laptop.intrinsics)

    prev_switch = None
    prev_ekf = None
    jumps_switch, jumps_ekf = [], []
    err_switch, err_ekf = [], []

    for k, (t, truth) in enumerate(trajectory(n, hand)):
        pa = laptop.intrinsics.project(laptop.world_to_cam(truth)) + rng.normal(0, noise, (21, 2))
        pb = None
        if vis[k]:
            pb = phone_true.intrinsics.project(phone_true.world_to_cam(truth)) \
                 + rng.normal(0, noise, (21, 2))

        # --- estimate-level switching (the old behaviour) -------------------
        if pb is not None:
            est = triangulate_hand([laptop, phone_belief], [pa, pb])
        else:
            centred = truth - truth.mean(0) + rng.normal(0, 0.004, (21, 3))
            res = solver_mono.solve(centred, pa)
            est = laptop.cam_to_world(res[0]) if res is not None else None
        if est is not None and np.all(np.isfinite(est)):
            if prev_switch is not None and vis[k] != vis[k - 1]:
                jumps_switch.append(np.linalg.norm(est - prev_switch, axis=1).mean())
            prev_switch = est
            err_switch.append(np.linalg.norm(est - truth, axis=1).mean())

        # --- measurement-level fusion (the new behaviour) -------------------
        ekf.predict(t)
        if not ekf.initialised:
            if pb is not None:
                seed = triangulate_hand([laptop, phone_belief], [pa, pb])
                if np.all(np.isfinite(seed)):
                    ekf.seed(seed, t)
            continue
        ekf.update_camera(laptop, pa, sigma_px=noise)
        if pb is not None:
            ekf.update_camera(phone_belief, pb, sigma_px=noise)
        else:
            # Only one view: depth is unobservable from pixels alone, so supply
            # a PnP fix against the known hand size as a 3D measurement.
            centred = truth - truth.mean(0) + rng.normal(0, 0.004, (21, 3))
            res = solver_mono.solve(centred, pa)
            if res is not None:
                ekf.update_position(laptop.cam_to_world(res[0]), sigma_m=0.015)
        cur = ekf.positions()
        if np.all(np.isfinite(cur)):
            ekf.apply_constraint(refine_with_skeleton(cur, bones, [], []), strength=0.4)
        cur = ekf.positions()
        if np.all(np.isfinite(cur)):
            if prev_ekf is not None and vis[k] != vis[k - 1]:
                jumps_ekf.append(np.linalg.norm(cur - prev_ekf, axis=1).mean())
            prev_ekf = cur
            err_ekf.append(np.linalg.norm(cur - truth, axis=1).mean())

    return (np.mean(jumps_switch) * 1000, np.max(jumps_switch) * 1000, np.mean(err_switch) * 1000,
            np.mean(jumps_ekf) * 1000, np.max(jumps_ekf) * 1000, np.mean(err_ekf) * 1000,
            len(jumps_switch))


def main():
    print("Phone view flickering in and out, hand moving continuously.")
    print("'jump' is the position change at a visibility transition -- the visible artifact.\n")
    print(f"{'rig scale':>10} | {'switching: mean/max jump':>26} | {'fusion: mean/max jump':>24}")
    print("-" * 70)

    for label, err in [("exact", 1.0), ("+10% off", 1.10), ("+25% off", 1.25)]:
        js, jsx, es, je, jex, ee, n = run(scale_error=err)
        print(f"{label:>10} | {js:>10.1f} / {jsx:>7.1f} mm | {je:>9.1f} / {jex:>7.1f} mm")

    print("\nTracking error over the same runs (mm):")
    print(f"{'rig scale':>10} | {'switching':>10} | {'fusion':>10}")
    print("-" * 38)
    for label, err in [("exact", 1.0), ("+10% off", 1.10), ("+25% off", 1.25)]:
        js, jsx, es, je, jex, ee, n = run(scale_error=err)
        print(f"{label:>10} | {es:>7.1f} mm | {ee:>7.1f} mm")


if __name__ == "__main__":
    main()

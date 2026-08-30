"""Calibrate the rig using your hand instead of a printed checkerboard.

The checkerboard version is more accurate and remains the right thing to do
eventually. But it needs a printer, a rigid backing, and a measured square, and
until that happens the phone contributes nothing at all -- so this exists to get
the second camera working today with no props whatsoever.

How it works. Both cameras already detect the same 21 hand landmarks, which
means every frame where both see your hand hands us 21 point correspondences
for free. Accumulate those over a few hundred frames while you move your hand
around, and there is enough information to recover the geometry between the two
cameras:

  1. The essential matrix, from the correspondences, gives the rotation and the
     *direction* of the translation between the cameras.
  2. That translation has no scale -- the same images would result from two
     cameras twice as far apart viewing a hand twice as large. Structure from
     motion always has this ambiguity, and it is why this needs a third step.
  3. Scale comes from your hand. MediaPipe estimates hand landmarks in real
     metres, so we know roughly how long your bones actually are. Reconstructing
     them with a unit baseline and comparing against that known length gives the
     factor that converts the whole rig to metres.

Accuracy is lower than a checkerboard: the landmarks are noisier than detected
corners, and the scale inherits whatever error MediaPipe's metric estimate has.
Expect it to be good enough to make the phone clearly worth having, and expect
the checkerboard to improve it later.

    python tools/calibrate_rig_hand.py --cam-a 0 --cam-b 2

Move your hand around a lot: near and far, left and right, high and low, palm
and back. Points clustered in one spot produce a degenerate solution that looks
fine and is badly wrong.

Keys:  c calibrate now   r reset   q quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import BONES, HandDetector
from handtrack.fusion import CameraRig, triangulate_hand
from handtrack.runtime import Interrupt
from handtrack.skeleton import HandModel
from handtrack.sources import describe_spec, open_source


def spatial_coverage(points: np.ndarray, w: int, h: int, grid: int = 6) -> float:
    """Fraction of a coarse image grid the samples have visited.

    A proxy for whether the correspondences are well spread. Low coverage means
    the hand stayed in one place, which makes the essential matrix
    ill-conditioned however many frames were collected.
    """
    if len(points) == 0:
        return 0.0
    gx = np.clip((points[:, 0] / w * grid).astype(int), 0, grid - 1)
    gy = np.clip((points[:, 1] / h * grid).astype(int), 0, grid - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(grid * grid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-a", default="0", help="Laptop camera")
    ap.add_argument("--cam-b", default="2", help="Phone camera (index, or 'record3d')")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov-a", type=float, default=60.0)
    ap.add_argument("--hfov-b", type=float, default=65.0)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--frames", type=int, default=240, help="Paired frames to collect")
    ap.add_argument("--out", default="config/calib/rig.json")
    args = ap.parse_args()

    print(f"A: {describe_spec(args.cam_a)}")
    print(f"B: {describe_spec(args.cam_b)}")
    # This tool cannot do anything useful with one camera, so a failure here is
    # fatal rather than something to degrade past -- but it should say why.
    cam_a = cam_b = None
    try:
        cam_a = open_source(args.cam_a, args.width, args.height, name="A")
        cam_b = open_source(args.cam_b, args.width, args.height, name="B")
    except Exception as exc:
        for c in (cam_a, cam_b):
            if c is not None:
                try: c.release()
                except Exception: pass
        print(f"\nCould not open both cameras:\n{exc}")
        print("\nThis tool needs BOTH cameras live at once. If the phone is the")
        print("problem, make sure its webcam app is actually streaming -- open Iriun")
        print("on the phone AND on the PC and check you can see the video there")
        print("before running this. Then confirm with:")
        print("  python tools/list_cameras.py")
        return 1
    print(f"  A {cam_a.width}x{cam_a.height}   B {cam_b.width}x{cam_b.height}\n")

    # Assumed intrinsics. They are the weakest link here, and the reason the
    # checkerboard route stays worth doing later.
    intr_a = PinholeCamera.from_fov(cam_a.width, cam_a.height, args.hfov_a, "laptop")
    intr_b = PinholeCamera.from_fov(cam_b.width, cam_b.height, args.hfov_b, "phone")

    det_a = HandDetector(args.model, num_hands=1)
    det_b = HandDetector(args.model, num_hands=1)
    model = HandModel(calibration_frames=60)

    pts_a: list[np.ndarray] = []
    pts_b: list[np.ndarray] = []
    la = lb = -1
    cur_a = cur_b = None
    win = "hand-based rig calibration - move your hand around BOTH views"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1500, 600)

    print("Move your hand all over both views: near, far, left, right, high, low.")
    print("Collecting... 'c' calibrates early, 'r' resets, 'q' quits.\n")

    interrupt = Interrupt().install()
    while True:
        if interrupt:
            print("\ninterrupted before calibrating")
            cam_a.release(); cam_b.release(); cv2.destroyAllWindows()
            return 1
        fa, fb = cam_a.read(), cam_b.read()
        if fa is None or fb is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cam_a.release(); cam_b.release(); cv2.destroyAllWindows()
                return 1
            continue
        fresh = False
        if fa.index != la:
            la = fa.index
            h = det_a.detect(fa.image, fa.timestamp, fa.index)
            cur_a = h[0] if h else None
            fresh = True
        if fb.index != lb:
            lb = fb.index
            h = det_b.detect(fb.image, fb.timestamp, fb.index)
            cur_b = h[0] if h else None
            fresh = True

        if fresh and cur_a is not None and cur_b is not None:
            pts_a.append(cur_a.pixels.copy())
            pts_b.append(cur_b.pixels.copy())
            model.observe(cur_a.world)

        va, vb = fa.image.copy(), fb.image.copy()
        for canvas, obs in ((va, cur_a), (vb, cur_b)):
            if obs is not None:
                for p in obs.pixels:
                    cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (90, 255, 140), -1)

        n = len(pts_a)
        cov = spatial_coverage(np.concatenate(pts_a), cam_a.width, cam_a.height) if pts_a else 0.0
        both = cur_a is not None and cur_b is not None
        status = (f"pairs {n}/{args.frames}   coverage {cov*100:.0f}%   "
                  + ("BOTH VIEWS" if both else "need the hand in BOTH views"))

        hh = max(va.shape[0], vb.shape[0])
        canvas = np.zeros((hh + 40, va.shape[1] + vb.shape[1], 3), np.uint8)
        canvas[40:40 + va.shape[0], :va.shape[1]] = va
        canvas[40:40 + vb.shape[0], va.shape[1]:] = vb
        cv2.putText(canvas, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (90, 255, 140) if both else (80, 130, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cam_a.release(); cam_b.release(); cv2.destroyAllWindows()
            print("aborted")
            return 1
        if key == ord("r"):
            pts_a.clear(); pts_b.clear()
            print("reset")
        if key == ord("c") or n >= args.frames:
            break

    cam_a.release(); cam_b.release()
    det_a.close(); det_b.close()
    cv2.destroyAllWindows()

    if len(pts_a) < 30:
        print(f"\nOnly {len(pts_a)} paired frames -- not enough. Make sure BOTH cameras")
        print("can see your hand at the same time, and try again.")
        return 1

    A = np.concatenate(pts_a).astype(np.float64)
    B = np.concatenate(pts_b).astype(np.float64)
    cov = spatial_coverage(A, cam_a.width, cam_a.height)
    print(f"\nCollected {len(pts_a)} paired frames ({len(A)} correspondences), coverage {cov*100:.0f}%")
    if cov < 0.25:
        print("  WARNING: the hand covered little of the frame. The solution may be")
        print("  poorly conditioned -- consider redoing this with more movement.")

    # Work in normalised camera coordinates so the two cameras may have
    # different intrinsics; findEssentialMat assumes a single shared matrix.
    na = cv2.undistortPoints(A.reshape(-1, 1, 2), intr_a.K, intr_a.dist).reshape(-1, 2)
    nb = cv2.undistortPoints(B.reshape(-1, 1, 2), intr_b.K, intr_b.dist).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC,
                                   prob=0.9999, threshold=0.003)
    if E is None or E.shape != (3, 3):
        print("Essential matrix estimation failed. Try again with more varied motion.")
        return 1
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"essential matrix: {inliers}/{len(na)} inliers ({100.0*inliers/len(na):.0f}%)")
    if inliers < 100:
        print("  too few inliers to trust; redo with more hand movement")
        return 1

    _, R, t, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=mask)
    t = t.ravel()
    t /= max(np.linalg.norm(t), 1e-12)     # direction only; scale comes next

    if not model.calibrated:
        print("Hand model never converged, so there is no known length to set the scale.")
        print("Hold your hand clearly in the laptop view for a few seconds and retry.")
        return 1

    # Resolve scale: reconstruct with a unit baseline, then compare the bones we
    # built against the bones we know are real.
    rig_a = CameraRig(intr_a, np.eye(3), np.zeros(3), "laptop")
    rig_b_unit = CameraRig(intr_b, R, t, "phone")
    ratios = []
    for pa, pb in zip(pts_a, pts_b):
        X = triangulate_hand([rig_a, rig_b_unit], [pa, pb])
        if not np.all(np.isfinite(X)):
            continue
        for k, (c, p) in enumerate(BONES):
            L = np.linalg.norm(X[c] - X[p])
            if L > 1e-9:
                ratios.append(model.bone_lengths[k] / L)
    if len(ratios) < 50:
        print("Could not resolve scale -- too few usable triangulations.")
        return 1

    scale = float(np.median(ratios))
    spread = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
    t_metric = t * scale
    baseline = float(np.linalg.norm(t_metric))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    print(f"scale factor: {scale:.4f}  (IQR {spread:.4f}, lower is more consistent)")
    print(f"\n  baseline: {baseline*100:.1f} cm")
    print(f"  angle:    {angle:.1f} deg")

    if not (0.02 < baseline < 1.5):
        print("\n  WARNING: implausible baseline. The scale step probably failed;")
        print("  the checkerboard route would be more reliable here.")
    if angle < 15:
        print("\n  WARNING: the cameras are nearly parallel, so depth will be weak.")
        print("  Angle the phone further toward your hands (aim for 40-50 deg).")

    # Report the residual as a reality check.
    rig_b = CameraRig(intr_b, R, t_metric, "phone")
    errs = []
    for pa, pb in zip(pts_a[:80], pts_b[:80]):
        X = triangulate_hand([rig_a, rig_b], [pa, pb])
        if not np.all(np.isfinite(X)):
            continue
        for k, (c, p) in enumerate(BONES):
            errs.append(abs(np.linalg.norm(X[c] - X[p]) - model.bone_lengths[k]))
    if errs:
        print(f"  bone-length residual: {np.mean(errs)*1000:.2f} mm mean "
              f"({np.percentile(errs,90)*1000:.2f} mm p90)")

    out = {
        "method": "hand-based (markerless)",
        "pairs": len(pts_a),
        "inlier_fraction": inliers / len(na),
        "coverage": cov,
        "scale_iqr": spread,
        "rms": {"a": float("nan"), "b": float("nan"), "stereo": float("nan")},
        "baseline_m": baseline,
        "angle_deg": angle,
        "cam_a": intr_a.to_dict(),
        "cam_b": intr_b.to_dict(),
        "R_b_from_a": R.tolist(),
        "T_b_from_a": t_metric.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")
    print("\nThe phone now contributes. Run:")
    print(f"  python run_fusion.py --laptop {args.cam_a} --phone {args.cam_b}")
    print(f"  python run_mouse.py  --laptop {args.cam_a} --phone {args.cam_b}")
    print("\nFor better accuracy later, redo it with tools/calibrate_rig.py and a")
    print("printed checkerboard -- that also measures the intrinsics properly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

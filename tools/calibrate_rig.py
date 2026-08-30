"""Calibrate the two-camera rig.

Triangulation is only as good as its knowledge of where the cameras are. This
tool measures that, in two stages:

  intrinsics  -- each camera's focal length, optical centre and lens distortion
  extrinsics  -- the rotation and translation between the two cameras

You need a printed checkerboard. Any standard one works; the default expects
9x6 *interior corners* (a 10x7 grid of squares). Print it, measure one square
with a ruler, and pass the real measurement with --square-mm, because that
measurement is what sets the absolute scale of everything downstream. Getting
it wrong by 5% makes every distance the system reports wrong by 5%.

Tape it to something flat and rigid -- a clipboard or a hardback book. A sagging
sheet of paper is the most common cause of a calibration that looks like it
succeeded but produces warped depth.

Usage:
  python tools/calibrate_rig.py --cam-a 0 --cam-b record3d --square-mm 25

Hold the board so BOTH cameras can see it, and move it around: near, far,
tilted, in each corner of the frame. The tool captures a pair each time both
views find the board and enough time has passed. Aim for 25-40 pairs covering
varied angles -- boards held flat-on in the centre every time produce a
confident-looking but badly conditioned solution.

Keys:  SPACE force-capture the current pair   c run calibration   q quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.sources import describe_spec, open_source

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(gray: np.ndarray, pattern: tuple[int, int]):
    ok, corners = cv2.findChessboardCorners(gray, pattern, FLAGS)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-a", default="0", help="Laptop camera index")
    ap.add_argument("--cam-b", default="record3d",
                    help="Phone source: 'record3d' for the iPhone over USB, or a camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, default=9, help="Interior corners across")
    ap.add_argument("--rows", type=int, default=6, help="Interior corners down")
    ap.add_argument("--square-mm", type=float, default=25.0, help="MEASURE THIS on your printout")
    ap.add_argument("--min-pairs", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.7, help="Min seconds between captures")
    ap.add_argument("--out", type=str, default="config/calib/rig.json")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm / 1000.0        # work in metres throughout

    print(f"Opening A: {describe_spec(args.cam_a)}")
    print(f"Opening B: {describe_spec(args.cam_b)}")
    cam_a = open_source(args.cam_a, args.width, args.height, name="A")
    cam_b = open_source(args.cam_b, args.width, args.height, name="B")
    print(f"  A: {cam_a.width}x{cam_a.height}    B: {cam_b.width}x{cam_b.height}")
    print(f"\nBoard: {args.cols}x{args.rows} interior corners, {args.square_mm} mm squares")
    print("Move the board around both views. SPACE forces a capture, 'c' calibrates, 'q' quits.\n")

    obj_points: list[np.ndarray] = []
    pts_a: list[np.ndarray] = []
    pts_b: list[np.ndarray] = []
    last_capture = 0.0
    win = "rig calibration - hold the board in BOTH views"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 600)

    while True:
        fa, fb = cam_a.read(), cam_b.read()
        if fa is None or fb is None:
            continue
        ga = cv2.cvtColor(fa.image, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(fb.image, cv2.COLOR_BGR2GRAY)
        ca, cb = find_corners(ga, pattern), find_corners(gb, pattern)

        va, vb = fa.image.copy(), fb.image.copy()
        if ca is not None:
            cv2.drawChessboardCorners(va, pattern, ca, True)
        if cb is not None:
            cv2.drawChessboardCorners(vb, pattern, cb, True)

        both = ca is not None and cb is not None
        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        forced = key == ord(" ")

        if both and (forced or now - last_capture > args.interval):
            obj_points.append(objp.copy())
            pts_a.append(ca)
            pts_b.append(cb)
            last_capture = now
            print(f"  captured pair {len(obj_points)}")

        status = f"pairs: {len(obj_points)}/{args.min_pairs}   " + \
                 ("BOTH VIEWS OK" if both else "board not visible in both")
        colour = (90, 255, 140) if both else (80, 130, 255)
        h = max(va.shape[0], vb.shape[0])
        canvas = np.zeros((h + 40, va.shape[1] + vb.shape[1], 3), np.uint8)
        canvas[40:40 + va.shape[0], :va.shape[1]] = va
        canvas[40:40 + vb.shape[0], va.shape[1]:] = vb
        cv2.putText(canvas, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        if key == ord("q"):
            print("aborted"); cam_a.release(); cam_b.release(); cv2.destroyAllWindows(); return 1
        if key == ord("c"):
            break

    cam_a.release(); cam_b.release(); cv2.destroyAllWindows()

    if len(obj_points) < 6:
        print(f"\nOnly {len(obj_points)} pairs -- far too few to calibrate. Try again.")
        return 1
    if len(obj_points) < args.min_pairs:
        print(f"\nWarning: only {len(obj_points)} pairs; results will be rough.")

    size_a = (cam_a.width, cam_a.height)
    size_b = (cam_b.width, cam_b.height)

    print(f"\nCalibrating intrinsics from {len(obj_points)} pairs...")
    rms_a, Ka, da, _, _ = cv2.calibrateCamera(obj_points, pts_a, size_a, None, None)
    rms_b, Kb, db, _, _ = cv2.calibrateCamera(obj_points, pts_b, size_b, None, None)
    print(f"  camera A reprojection RMS: {rms_a:.3f} px  (want < 0.6)")
    print(f"  camera B reprojection RMS: {rms_b:.3f} px  (want < 0.6)")

    print("Calibrating extrinsics...")
    rms_stereo, Ka, da, Kb, db, R, T, E, F = cv2.stereoCalibrate(
        obj_points, pts_a, pts_b, Ka, da, Kb, db, size_a,
        criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)

    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"  stereo reprojection RMS: {rms_stereo:.3f} px  (want < 1.0)")
    print(f"\n  baseline (camera separation): {baseline*100:.1f} cm")
    print(f"  angle between cameras:        {angle:.1f} deg")

    if rms_stereo > 1.5:
        print("\n  WARNING: high reprojection error. Common causes: a non-flat board,")
        print("  a wrong --square-mm, motion blur during capture, or too few varied poses.")
    if angle < 20:
        print("\n  WARNING: cameras are nearly parallel. Depth accuracy will be poor;")
        print("  increase the phone's angle toward 40-50 degrees.")

    out = {
        "square_mm": args.square_mm,
        "pattern": [args.cols, args.rows],
        "pairs": len(obj_points),
        "rms": {"a": rms_a, "b": rms_b, "stereo": rms_stereo},
        "baseline_m": baseline,
        "angle_deg": angle,
        "cam_a": {"K": Ka.tolist(), "dist": da.ravel().tolist(),
                  "width": size_a[0], "height": size_a[1], "name": "laptop"},
        "cam_b": {"K": Kb.tolist(), "dist": db.ravel().tolist(),
                  "width": size_b[0], "height": size_b[1], "name": "phone"},
        # Pose of camera B relative to camera A. Camera A defines the world frame.
        "R_b_from_a": R.tolist(),
        "T_b_from_a": T.ravel().tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

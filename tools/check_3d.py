"""Is the reconstructed hand physically real?

Everything downstream -- pointing, pinching, screen calibration -- assumes the
3D hand is metrically correct. If it is not, all of them fail together and each
looks like its own separate bug. This checks the 3D directly against facts we
already know about hands, before any of that machinery is involved.

Three independent checks, none of which need a calibration:

  distance    a hand held in front of the phone should be 25-60 cm away
  span        wrist to middle knuckle is about 9 cm on an adult hand, and the
              full hand about 18 cm. These are known quantities, so a
              reconstruction that disagrees is wrong regardless of how
              plausible it looks
  flatness    a hand is not flat, but neither is it 40 cm deep. If the depth
              spread across the landmarks is wildly larger than the hand, depth
              is being sampled from the wrong place

The most likely way to fail these is a mismatch between the colour image and the
depth map. Depth is looked up by scaling landmark pixels into depth-map
coordinates, which is only valid when both cover the SAME field of view. If the
colour stream is 16:9 and the depth stream is 4:3, that scaling silently reads
depth from the wrong part of the scene -- landmarks all get a depth, so nothing
appears broken, but every one of them is wrong.

    python tools/check_3d.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import (INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP,
                              THUMB_TIP, WRIST)
from handtrack.fusion import CameraRig
from handtrack.runtime import Interrupt
from handtrack.sources.handbridge_source import DEFAULT_PORT, HandBridgeSource
from run_hand import back_project

# Adult hand, from anthropometric tables. Generous ranges: this is a sanity
# check, not a measurement.
PALM_CM = (7.5, 11.5)      # wrist to middle knuckle
WIDTH_CM = (7.0, 11.0)     # index knuckle to little knuckle
DIST_CM = (20.0, 70.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    print("Hold your hand open, palm toward the phone, about 40 cm away.\n")
    src = HandBridgeSource(port=args.port)
    try:
        src.start(timeout=60.0, on_wait=lambda n, l: print(f"  waiting {l:.0f}s", flush=True))
    except RuntimeError as exc:
        print(exc)
        return 1

    f0 = src.read()
    print(f"stream: {f0.width}x{f0.height}   "
          f"fx={f0.intrinsics[0,0]:.1f} fy={f0.intrinsics[1,1]:.1f} "
          f"cx={f0.intrinsics[0,2]:.1f} cy={f0.intrinsics[1,2]:.1f}")

    # A principal point that is not near the middle of the image is the
    # signature of intrinsics belonging to a differently-sized buffer.
    off_x = f0.intrinsics[0, 2] / f0.width
    off_y = f0.intrinsics[1, 2] / f0.height
    if not (0.35 < off_x < 0.65 and 0.35 < off_y < 0.65):
        print(f"  !! principal point sits at ({off_x*100:.0f}%, {off_y*100:.0f}%) of the")
        print("     image rather than near the centre. The intrinsics probably")
        print("     describe a different buffer size than the landmarks do.")

    rig = CameraRig(PinholeCamera(f0.intrinsics, np.zeros(5), f0.width, f0.height),
                    np.eye(3), np.zeros(3), "phone")

    interrupt = Interrupt().install()
    import time
    end = time.time() + args.seconds
    dists, palms, widths, spreads = [], [], [], []
    last = -1
    n_hand = n_depth = 0

    while time.time() < end and not interrupt:
        f = src.read()
        if f is None or f.index == last:
            time.sleep(0.003)
            continue
        last = f.index
        if not f.has_hand:
            continue
        n_hand += 1
        if int(np.isfinite(f.depth).sum()) < 10:
            continue
        n_depth += 1

        p = back_project(rig, f.pixels, f.depth)
        need = [WRIST, MIDDLE_MCP, INDEX_MCP, PINKY_MCP]
        if not np.all(np.isfinite(p[need])):
            continue

        dists.append(np.nanmedian(p[:, 2]) * 100)
        palms.append(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]) * 100)
        widths.append(np.linalg.norm(p[INDEX_MCP] - p[PINKY_MCP]) * 100)
        good = np.isfinite(p[:, 2])
        spreads.append((p[good, 2].max() - p[good, 2].min()) * 100)

    src.release()

    if not dists:
        print("\nNo usable frames. Hold an open hand ~40 cm from the phone and retry.")
        return 1

    def report(name, values, lo, hi, unit="cm"):
        v = float(np.median(values))
        ok = lo <= v <= hi
        flag = "OK  " if ok else "BAD "
        print(f"  [{flag}] {name:<22} {v:6.1f} {unit}   (expect {lo:.0f}-{hi:.0f})")
        return ok

    print(f"\nmeasured over {len(dists)} frames "
          f"({n_hand} with a hand, {n_depth} of those with depth)\n")
    ok1 = report("distance to hand", dists, *DIST_CM)
    ok2 = report("palm length", palms, *PALM_CM)
    ok3 = report("hand width", widths, *WIDTH_CM)
    ok4 = report("depth spread", spreads, 0, 25)

    print()
    if all((ok1, ok2, ok3, ok4)):
        print("3D reconstruction is physically sensible. If pointing still misbehaves,")
        print("the fault is in the screen calibration or the pointing mode, not here.")
        return 0

    print("3D reconstruction is WRONG, which explains shaky tracking, failing")
    print("pinches and unusable pointing all at once -- they all read from this.")
    print()
    if not (ok2 and ok3):
        scale = float(np.median(palms)) / 9.0
        print(f"  The hand comes out about {scale:.2f}x its real size.")
        if abs(scale - 1) > 0.15:
            print("  A wrong scale means the depth values do not belong to the pixels")
            print("  they are paired with -- the usual cause is the colour and depth")
            print("  streams covering different fields of view.")
    if not ok4:
        print("  The depth spread across the hand is far too large, so some landmarks")
        print("  are reading depth from the background rather than from the hand.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

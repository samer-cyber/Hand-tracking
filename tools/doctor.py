"""Check the whole chain and report exactly what is blocking stereo tracking.

Run this first whenever something does not work. It walks the dependency chain
in order -- cameras respond, both see your hand at the same time, a rig
calibration exists and is plausible -- and stops at the first thing that is
actually broken, rather than leaving you to infer it from a crash.

    python tools/doctor.py --laptop 0 --phone 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.capture import CameraOpenError
from handtrack.detect import HandDetector
from handtrack.runtime import Interrupt
from handtrack.sources import describe_spec, open_source

OK, BAD, WARN = "  [ok]   ", " [FAIL]  ", " [warn]  "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", default="0")
    ap.add_argument("--phone", default="2")
    ap.add_argument("--rig", default="config/calib/rig.json")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    args = ap.parse_args()

    interrupt = Interrupt().install()
    print("Hand-tracking doctor\n" + "=" * 56)

    # 1 -- model file
    if not os.path.exists(args.model):
        print(BAD + f"missing model file {args.model}")
        return 1
    print(OK + "hand landmark model present")

    # 2 -- cameras open
    sources = {}
    for label, spec in (("laptop", args.laptop), ("phone", args.phone)):
        if str(spec).lower() == "none":
            continue
        try:
            sources[label] = open_source(spec, 1280, 720, name=label)
            s = sources[label]
            print(OK + f"{label}: {describe_spec(spec)} -> {s.width}x{s.height}")
        except CameraOpenError as exc:
            print(BAD + f"{label}: {describe_spec(spec)} did not respond")
            print("         " + str(exc).splitlines()[0])
            if label == "phone":
                print("         The phone webcam app is almost certainly not streaming.")
                print("         Open Iriun on BOTH the phone and the PC, confirm you can")
                print("         see live video in the PC app, then re-run this.")
        except Exception as exc:
            print(BAD + f"{label}: {exc}")

    if not sources:
        print("\nNo cameras available -- nothing else can be checked.")
        return 1

    # 3 -- do they deliver frames, and can each see a hand?
    print(f"\nWatching for {args.seconds:.0f}s -- hold ONE hand where BOTH cameras see it.\n")
    detectors = {k: HandDetector(args.model, num_hands=1) for k in sources}
    seen = {k: 0 for k in sources}
    frames = {k: 0 for k in sources}
    last = {k: -1 for k in sources}
    both = 0
    total = 0
    t_end = time.time() + args.seconds

    while time.time() < t_end and not interrupt:
        got = {}
        for k, src in sources.items():
            f = src.read()
            if f is None or f.index == last[k]:
                continue
            last[k] = f.index
            frames[k] += 1
            hands = detectors[k].detect(f.image, f.timestamp, f.index)
            got[k] = bool(hands)
            if hands:
                seen[k] += 1
        if got:
            total += 1
            if len(got) == len(sources) and all(got.values()):
                both += 1

    for k in sources:
        rate = 100.0 * seen[k] / max(frames[k], 1)
        fps = sources[k].measured_fps
        line = f"{k}: {frames[k]} frames ({fps:.1f} fps), hand seen in {rate:.0f}%"
        if frames[k] == 0:
            print(BAD + line + "  -- delivering NO frames")
        elif rate < 20:
            print(WARN + line + "  -- rarely sees your hand")
        else:
            print(OK + line)

    if len(sources) == 2:
        pct = 100.0 * both / max(total, 1)
        if pct < 15:
            print(BAD + f"both cameras saw the hand together in only {pct:.0f}% of frames")
            print("         Stereo needs BOTH views simultaneously. Aim the phone at the")
            print("         same space your hand occupies in front of the laptop, and")
            print("         check the overlap before calibrating.")
        else:
            print(OK + f"both cameras saw the hand together in {pct:.0f}% of frames")

    # 4 -- rig calibration
    print()
    if not os.path.exists(args.rig):
        print(WARN + f"no rig calibration at {args.rig}")
        print("         Without it the phone CANNOT contribute and everything falls")
        print("         back to laptop-only mono tracking. This is the usual reason")
        print("         the phone appears to be ignored. Fix with:")
        print(f"           python tools/calibrate_rig_hand.py --cam-a {args.laptop} --cam-b {args.phone}")
    else:
        try:
            meta = json.load(open(args.rig))
            ang, base = meta.get("angle_deg", float("nan")), meta.get("baseline_m", float("nan"))
            print(OK + f"rig calibration present: {ang:.1f} deg, baseline {base*100:.1f} cm "
                       f"({meta.get('method', 'checkerboard')})")
            if not (0.02 < base < 1.5):
                print(WARN + "  baseline is implausible -- recalibrate")
            if ang < 15:
                print(WARN + "  cameras nearly parallel -- depth will be weak; angle the phone more")
        except Exception as exc:
            print(BAD + f"rig calibration unreadable: {exc}")

    if os.path.exists("config/calib/hand_model.npz"):
        print(OK + "saved hand model present")
    else:
        print(WARN + "no saved hand model (it will be learned at runtime; press 's' to save)")

    for s in sources.values():
        s.release()
    for d in detectors.values():
        d.close()

    print("\n" + "=" * 56)
    ready = len(sources) == 2 and os.path.exists(args.rig)
    if ready:
        print("Ready for stereo. Run:")
        print(f"  python run_fusion.py --laptop {args.laptop} --phone {args.phone}")
        print("  (the HUD 'mode' line should read 'stereo')")
    else:
        print("Not ready for stereo yet -- fix the items marked FAIL/warn above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

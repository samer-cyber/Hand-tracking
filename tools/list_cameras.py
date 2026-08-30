"""Enumerate cameras safely, isolating each probe in its own process.

Blind index probing is genuinely dangerous on Windows. A virtual camera whose
source is absent -- Iriun with the phone not streaming, for instance -- makes
`cv2.VideoCapture(index)` block forever rather than fail. That wedges the tool,
and the stuck process keeps a handle on the driver, which then makes every
later run fail to open *any* camera. The symptom is "it worked, then nothing
worked", and the cause is a leaked handle rather than anything in the pipeline.

So each probe runs in a separate short-lived process with a hard timeout. If a
device hangs, that child is killed and its handle dies with it, and the parent
reports the device as unresponsive and moves on.

  python tools/list_cameras.py            # quick scan
  python tools/list_cameras.py --measure  # also time real frame arrivals
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Runs in the child process: open one camera, describe it, print JSON, exit.
PROBE = r"""
import sys, json, time
import cv2, numpy as np
idx, backend, measure = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
out = {"opened": False}
cap = cv2.VideoCapture(idx, backend)
if cap.isOpened():
    ok, frame = cap.read()
    if ok and frame is not None:
        out["opened"] = True
        out["w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out["h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
        b, g, r = (frame[:, :, i].astype(np.int16) for i in range(3))
        out["chan_diff"] = float((np.abs(b-g).mean() + np.abs(g-r).mean()) / 2)
        out["mean"] = float(frame.mean())
        modes = []
        for (rw, rh) in [(1920,1080), (1280,720), (640,480)]:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (rw, rh):
                modes.append(f"{rw}x{rh}")
        out["modes"] = modes
        if measure:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(10): cap.read()
            times = []
            end = time.perf_counter() + 3.0
            while time.perf_counter() < end:
                if cap.read()[0]: times.append(time.perf_counter())
            if len(times) > 5:
                iv = np.diff(times)
                out["meas_fps"] = float(1.0/np.median(iv))
                out["iv_med"] = float(np.median(iv)*1000)
                out["iv_p95"] = float(np.percentile(iv,95)*1000)
cap.release()
print(json.dumps(out))
"""

BACKENDS = [(1400, "MSMF"), (700, "DSHOW")]     # cv2.CAP_MSMF, cv2.CAP_DSHOW


def probe(index: int, backend: int, measure: bool, timeout: float):
    """Probe one camera in a child process. Returns a dict, or None on hang."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE, str(index), str(backend), "1" if measure else "0"],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None                              # child already killed
    for line in proc.stdout.strip().splitlines()[::-1]:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"opened": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=4)
    ap.add_argument("--measure", action="store_true", help="Time real frame arrivals (3 s each)")
    ap.add_argument("--timeout", type=float, default=12.0, help="Seconds before a device is called hung")
    args = ap.parse_args()

    timeout = args.timeout + (4.0 if args.measure else 0.0)
    print("Scanning cameras (each probe is isolated, so a hung device cannot wedge this)\n")

    hung, found = [], 0
    for backend, bname in BACKENDS:
        print(f"=== {bname} ===")
        any_here = False
        for i in range(args.max_index):
            info = probe(i, backend, args.measure, timeout)

            if info is None:
                hung.append((i, bname))
                print(f"  index {i}: DID NOT RESPOND within {timeout:.0f}s -- skipped")
                print("           a virtual camera with no active source does this")
                any_here = True
                continue
            if not info.get("opened"):
                continue

            any_here = True
            found += 1
            kind = "grayscale/IR" if info["chan_diff"] < 2.0 else "colour"
            print(f"  index {i}: {info['w']}x{info['h']} @ {info['fps']:.0f} declared, {kind}, "
                  f"brightness {info['mean']:.0f}")
            if info["mean"] < 5:
                print("           returns black -- likely a Windows Hello IR sensor")
            if info.get("modes"):
                print(f"           modes: {', '.join(info['modes'])}")
            if "meas_fps" in info:
                print(f"           measured {info['meas_fps']:.1f} fps "
                      f"(interval {info['iv_med']:.1f} ms, p95 {info['iv_p95']:.1f} ms)")
                if info["iv_p95"] > 2.5 * info["iv_med"]:
                    print("           uneven delivery -- buffering, or a busy USB bus")
        if not any_here:
            print("  none")
        print()

    if hung:
        print("Devices that hung on open:")
        for i, b in hung:
            print(f"  index {i} [{b}]")
        print("\n  This is normal for a phone-webcam driver (Iriun, EpocCam, DroidCam)")
        print("  when the phone is not currently streaming. Start the phone app first,")
        print("  then re-run. The probes were killed, so no handles were leaked.")

    print(f"\n{found} working camera entries (each physical device appears once per backend).")
    print("Prefer the MSMF index unless it misbehaves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

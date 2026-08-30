"""Headless benchmark of the full fusion pipeline.

Runs whatever streams are available for a fixed number of frames and reports
which solver modes were used and how well each performed. No window, no
interaction -- so it can be run over SSH, or compared between configurations
without judging by feel.

Hold a hand in view, and hold it still for the last few seconds so the jitter
figure has something to measure.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.cameras import PinholeCamera
from handtrack.detect import INDEX_TIP, HandDetector
from handtrack.filters import OneEuroFilter, VelocityPredictor
from handtrack.fusion import CameraRig, load_rig
from handtrack.metrics import JitterMeter, Rolling, StageTimer, TrackingHealth
from handtrack.pipeline import HandSolver
from handtrack.skeleton import HandModel
from handtrack.sources import describe_spec, open_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", default="0")
    ap.add_argument("--phone", default="none")
    ap.add_argument("--rig", default="config/calib/rig.json")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    args = ap.parse_args()

    use_a = str(args.laptop).lower() != "none"
    use_b = str(args.phone).lower() != "none"
    src_a = open_source(args.laptop, args.width, args.height, name="laptop") if use_a else None
    src_b = None
    if use_b:
        try:
            src_b = open_source(args.phone, args.width, args.height, name="phone")
        except Exception as exc:
            print(f"phone unavailable: {exc}")
            use_b = False

    has_depth = use_b and str(args.phone).lower() in ("record3d", "phone", "iphone")

    rig_a = rig_b = None
    if use_a and use_b and os.path.exists(args.rig):
        rig_a, rig_b, meta = load_rig(args.rig)
        print(f"rig: {meta['angle_deg']:.1f} deg, {meta['baseline_m']*100:.1f} cm")
    elif use_b and has_depth:
        rig_b = CameraRig(PinholeCamera.from_fov(src_b.width, src_b.height, args.hfov, "phone"),
                          np.eye(3), np.zeros(3), "phone")
    elif use_a:
        rig_a = CameraRig(PinholeCamera.from_fov(src_a.width, src_a.height, args.hfov, "laptop"),
                          np.eye(3), np.zeros(3), "laptop")

    det_a = HandDetector(args.model, num_hands=1) if use_a else None
    det_b = HandDetector(args.model, num_hands=1) if use_b else None
    model = HandModel(calibration_frames=60)
    model.load("config/calib/hand_model.npz")
    solver = HandSolver(rig_a, rig_b, model)
    filt = OneEuroFilter(1.5, 0.06)
    pred = VelocityPredictor(lead_seconds=0.03, max_step=0.08)

    timer, e2e, loop = StageTimer(args.frames), Rolling(args.frames), Rolling(args.frames)
    j3 = JitterMeter(still_threshold_mps=0.05, scale=1000.0)
    health = TrackingHealth(args.frames)
    modes: dict[str, int] = {}
    bone_res, reproj, ndepth = Rolling(args.frames), Rolling(args.frames), Rolling(args.frames)

    print(f"collecting {args.frames} frames -- hold a hand in view, still at the end\n")
    seen, la, lb = 0, -1, -1
    t_last = time.perf_counter()

    while seen < args.frames:
        fa = src_a.read() if src_a else None
        fb = src_b.read() if src_b else None
        new = (fa is not None and fa.index != la) or (fb is not None and fb.index != lb)
        if not new:
            time.sleep(0.001)
            continue
        seen += 1
        now = time.perf_counter()
        loop.add(1.0 / max(now - t_last, 1e-4)); t_last = now
        timer.start()

        px_a = world_a = px_b = world_b = None
        if fa is not None and fa.index != la:
            la = fa.index
            h = det_a.detect(fa.image, fa.timestamp, fa.index)
            if h:
                px_a, world_a = h[0].pixels, h[0].world
        if fb is not None and fb.index != lb:
            lb = fb.index
            h = det_b.detect(fb.image, fb.timestamp, fb.index)
            if h:
                px_b, world_b = h[0].pixels, h[0].world
        timer.mark("detect")

        depth = None
        if has_depth and px_b is not None and fb is not None:
            from handtrack.sources.record3d_source import sample_depth
            depth = sample_depth(fb, px_b)
            ndepth.add(int(np.isfinite(depth).sum()))
        timer.mark("depth")

        ts = max([f.timestamp for f in (fa, fb) if f is not None])
        res = solver.solve(px_a, px_b, depth, ts, world_a=world_a, world_b=world_b)
        timer.mark("solve")

        modes[res.mode] = modes.get(res.mode, 0) + 1
        health.update(res.ok)
        if np.isfinite(res.bone_residual_mm):
            bone_res.add(res.bone_residual_mm)
        if np.isfinite(res.reprojection_px):
            reproj.add(res.reprojection_px)
        if res.ok:
            f = pred(filt(res.points, ts), ts, lead=0.03)
            j3.update(f[INDEX_TIP], ts)
        timer.mark("filter")
        e2e.add((time.perf_counter() - ts) * 1000.0)

        if seen % 50 == 0:
            print(f"  {seen}/{args.frames}  solved {health.rate*100:.0f}%  "
                  f"calib {model.progress*100:.0f}%")

    print(f"\n=== results ===")
    print(f"loop fps        {loop.median:6.1f}")
    print(f"latency         {e2e.median:6.1f} ms   p95 {e2e.p95:6.1f} ms")
    for name, r in timer.stages.items():
        print(f"  {name:<12}      {r.median:6.2f} ms")
    print(f"solved          {health.rate*100:6.1f} %   dropouts {health.dropouts}")
    if reproj.buf:
        print(f"reprojection    {reproj.median:6.2f} px")
    if bone_res.buf:
        print(f"bone residual   {bone_res.median:6.2f} mm")
    if ndepth.buf:
        print(f"depth samples   {ndepth.median:6.1f} / 21 landmarks")
    print(f"3D jitter       {j3.jitter:6.2f} mm  [{j3.samples} still samples]")
    total = sum(modes.values()) or 1
    print("modes:")
    for m, c in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<18} {c*100.0/total:5.1f} %")

    for s in (src_a, src_b):
        if s:
            s.release()
    for d in (det_a, det_b):
        if d:
            d.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

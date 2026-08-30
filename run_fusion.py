"""The full system: laptop camera + iPhone colour + iPhone TrueDepth.

Runs every stream that is available and fuses them into one metric 3D hand,
degrading gracefully as streams come and go. The HUD reports which method
produced each frame, so you can watch it switch modes as your hand leaves one
camera's view.

Three ways to run it, in increasing order of setup effort:

  1. Phone only -- no calibration at all.
     Full metric 3D straight away, because the depth sensor measures distance
     directly and needs no second viewpoint to do it.

         python run_fusion.py --laptop none --phone record3d

  2. Laptop only -- no calibration, no phone.
         python run_fusion.py --phone none

  3. Both, calibrated. Best accuracy and the widest coverage.
         python tools/calibrate_rig.py --cam-a 0 --cam-b record3d --square-mm 25
         python run_fusion.py --laptop 0 --phone record3d

Keys:  q quit   r cycle render   p prediction   h HUD
       c recalibrate hand   f freeze hand model   s save hand calibration
       x REALIGN the phone to the laptop
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handtrack.cameras import PinholeCamera
from handtrack.detect import INDEX_TIP, HandDetector
from handtrack.filters import OneEuroFilter, VelocityPredictor
from handtrack.fusion import CameraRig, adapt_rig_to_sources, load_rig
from handtrack.metrics import JitterMeter, Rolling, StageTimer, TrackingHealth
from handtrack.pipeline import MODE_NONE, HandSolver
from handtrack.realign import PhoneRealigner
from handtrack.runtime import Interrupt
from handtrack.skeleton import HandModel
from handtrack.sources import describe_spec, open_source
from handtrack.viz import draw_hand, draw_hud

CALIB_PATH = "config/calib/hand_model.npz"


def save_rig(path, rig_a, rig_b, report):
    """Persist a rig whose phone pose was corrected at runtime."""
    import json
    data = {
        "method": "hand-based + runtime realign",
        "baseline_m": report["baseline_cm"] / 100.0,
        "angle_deg": report["angle_deg"],
        "realign_residual_px": report["residual_px"],
        "rms": {"stereo": float("nan")},
        "cam_a": rig_a.intrinsics.to_dict(),
        "cam_b": rig_b.intrinsics.to_dict(),
        "R_b_from_a": rig_b.R.tolist(),
        "T_b_from_a": rig_b.t.tolist(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)




RENDER_MODES = ["both", "processed", "raw"]


class View:
    """One camera stream plus its detector and image-space velocity."""

    def __init__(self, source, detector: HandDetector, name: str):
        self.source = source
        self.detector = detector
        self.name = name
        self.pixels: np.ndarray | None = None
        self.world: np.ndarray | None = None
        self.timestamp = 0.0
        self.velocity = np.zeros((21, 2))
        self.frame = None
        self.frame_timestamp = 0.0
        self.last_index = -1
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None

    def poll(self) -> bool:
        """Detect on the newest frame, if there is one. True if it is new."""
        frame = self.source.read()
        if frame is None or frame.index == self.last_index:
            return False
        self.last_index = frame.index
        self.frame = frame
        # Kept separately from `timestamp`, which only advances when a hand is
        # actually found. Using the detection timestamp for latency made a long
        # spell with no hand look like tens of seconds of lag.
        self.frame_timestamp = frame.timestamp
        hands = self.detector.detect(frame.image, frame.timestamp, frame.index)
        if not hands:
            self.pixels = self.world = None
            self._prev = self._prev_t = None
            return True
        obs = hands[0]
        if self._prev is not None and self._prev_t is not None:
            dt = frame.timestamp - self._prev_t
            if 1e-4 < dt < 0.25:
                self.velocity = 0.6 * self.velocity + 0.4 * (obs.pixels - self._prev) / dt
        self._prev, self._prev_t = obs.pixels.copy(), frame.timestamp
        self.pixels, self.world = obs.pixels, obs.world
        self.timestamp = frame.timestamp
        return True

    def pixels_at(self, t: float, max_shift: float = 60.0) -> np.ndarray | None:
        """Landmarks extrapolated to a common instant, for cross-view fusion."""
        if self.pixels is None:
            return None
        dt = t - self.timestamp
        if abs(dt) > 0.1:
            return None
        return self.pixels + np.clip(self.velocity * dt, -max_shift, max_shift)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laptop", default="0", help="Camera index, or 'none'")
    ap.add_argument("--phone", default="record3d", help="'record3d', a camera index, or 'none'")
    ap.add_argument("--rig", default="config/calib/rig.json")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--min-cutoff", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.06)
    ap.add_argument("--no-depth", action="store_true", help="Ignore the depth stream, for A/B")
    args = ap.parse_args()

    use_laptop = str(args.laptop).lower() != "none"
    use_phone = str(args.phone).lower() != "none"
    if not (use_laptop or use_phone):
        print("Nothing to do: both --laptop and --phone are 'none'.")
        return 1

    # --- open whatever streams were requested -------------------------------
    src_a = src_b = None
    if use_laptop:
        print(f"laptop: opening {describe_spec(args.laptop)}...")
        src_a = open_source(args.laptop, args.width, args.height, name="laptop")
        print(f"  {src_a.width}x{src_a.height}")
    if use_phone:
        print(f"phone:  opening {describe_spec(args.phone)}...")
        try:
            src_b = open_source(args.phone, args.width, args.height, name="phone")
            print(f"  {src_b.width}x{src_b.height}")
            if hasattr(src_b, "device_type"):
                print(f"  device type {src_b.device_type} (0 = TrueDepth front camera)")
        except Exception as exc:
            print(f"  could not open the phone:\n{exc}\n  continuing without it.")
            src_b, use_phone = None, False

    has_depth = use_phone and hasattr(src_b, "read") and not args.no_depth and \
        str(args.phone).lower() in ("record3d", "phone", "iphone")

    # --- work out the geometry ----------------------------------------------
    rig_a = rig_b = None
    world_frame = "none"
    if use_laptop and use_phone and os.path.exists(args.rig):
        rig_a, rig_b, meta = load_rig(args.rig)
        world_frame = "laptop (calibrated rig)"
        method = meta.get("method", "checkerboard")
        rms = meta.get("rms", {}).get("stereo", float("nan"))
        quality = f"RMS {rms:.2f} px" if rms == rms else f"{method}"
        print(f"rig: baseline {meta['baseline_m']*100:.1f} cm, "
              f"angle {meta['angle_deg']:.1f} deg, {quality}")
        adapt_rig_to_sources(rig_a, rig_b, src_a, src_b)
    elif use_phone and has_depth:
        # No calibration needed: the depth sensor measures distance directly, so
        # the phone's own frame can serve as the world frame.
        f = src_b.read()
        K = f.intrinsics if f is not None and np.any(f.intrinsics != np.eye(3)) else None
        intr = (PinholeCamera(K, np.zeros(5), src_b.width, src_b.height, "phone") if K is not None
                else PinholeCamera.from_fov(src_b.width, src_b.height, args.hfov, "phone"))
        rig_b = CameraRig(intr, np.eye(3), np.zeros(3), "phone")
        world_frame = "phone (uncalibrated, depth-driven)"
        if use_laptop:
            print("no rig calibration found -- using the phone alone.")
            print("  run tools/calibrate_rig.py to bring the laptop camera in too.")
    elif use_laptop:
        intr = PinholeCamera.from_fov(src_a.width, src_a.height, args.hfov, "laptop")
        rig_a = CameraRig(intr, np.eye(3), np.zeros(3), "laptop")
        world_frame = "laptop (assumed FOV)"
        if use_phone:
            # A second colour camera with no extrinsics is dead weight: without
            # knowing where it sits we cannot triangulate, so it cannot help.
            print(f"\n  !! The phone is connected but there is no rig calibration at {args.rig}.")
            print("     Without it the two views cannot be related, so the phone")
            print("     contributes nothing and this falls back to laptop-only mono")
            print("     tracking (roughly 12 mm, versus about 1.2 mm calibrated).")
            print("     Fix it with (no checkerboard needed):")
            print(f"       python tools/calibrate_rig_hand.py --cam-a {args.laptop} "
                  f"--cam-b {args.phone}\n")
    else:
        print("The phone is connected but has no depth, and there is no rig calibration.")
        return 1
    print(f"world frame: {world_frame}\n")

    if use_laptop and rig_a is None:
        # Phone-only world frame; the laptop cannot contribute without extrinsics.
        src_a.release()
        src_a, use_laptop = None, False

    # --- assemble the pipeline ----------------------------------------------
    view_a = View(src_a, HandDetector(args.model, num_hands=1), "laptop") if use_laptop else None
    view_b = View(src_b, HandDetector(args.model, num_hands=1), "phone") if use_phone else None

    model = HandModel(calibration_frames=90)
    if model.load(CALIB_PATH):
        print(f"loaded hand calibration from {CALIB_PATH}")
    solver = HandSolver(rig_a, rig_b, model)

    filt = OneEuroFilter(args.min_cutoff, args.beta)
    pred = VelocityPredictor(lead_seconds=0.030, max_step=0.08)
    timer, e2e, loopfps = StageTimer(), Rolling(120), Rolling(120)
    reproj, bone_res, depth_dis, skew = Rolling(120), Rolling(120), Rolling(120), Rolling(120)
    j3 = JitterMeter(still_threshold_mps=0.05, scale=1000.0)
    health = TrackingHealth()
    mode_counts: dict[str, int] = {}

    render_mode, use_pred, show_hud = 0, True, True
    _hand_saved = model.calibrated
    realigner = (PhoneRealigner(rig_a, rig_b, model)
                 if (rig_a is not None and rig_b is not None) else None)
    points = None
    last_loop = time.perf_counter()

    interrupt = Interrupt().install()
    done = {"shut": False}

    def shutdown():
        if done["shut"]:
            return
        done["shut"] = True
        for v in (view_a, view_b):
            if v is not None:
                try: v.source.release()
                except Exception: pass
                try: v.detector.close()
                except Exception: pass
        cv2.destroyAllWindows()

    atexit.register(shutdown)

    win = "Hand tracking - fusion"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1500, 640)

    while True:
        if interrupt:
            break
        new = False
        if view_a is not None:
            new |= view_a.poll()
        if view_b is not None:
            new |= view_b.poll()
        if not new:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        now = time.perf_counter()
        loopfps.add(1.0 / max(now - last_loop, 1e-4))
        last_loop = now
        timer.start()
        timer.mark("detect")

        stamps = [v.timestamp for v in (view_a, view_b) if v is not None and v.pixels is not None]
        target = max(stamps) if stamps else now
        if view_a is not None and view_b is not None and len(stamps) == 2:
            skew.add(abs(view_a.timestamp - view_b.timestamp) * 1000.0)

        px_a = view_a.pixels_at(target) if view_a is not None else None
        px_b = view_b.pixels_at(target) if view_b is not None else None

        depth = None
        if has_depth and px_b is not None and view_b.frame is not None:
            from handtrack.sources.record3d_source import sample_depth
            depth = sample_depth(view_b.frame, px_b)
        timer.mark("depth")

        wa = view_a.world if view_a else None
        wb = view_b.world if view_b else None
        res = solver.solve_fused(px_a, px_b, depth, target, world_a=wa, world_b=wb)
        timer.mark("solve")

        if realigner is not None and realigner.active:
            if realigner.observe(px_a, px_b, wa):
                out = realigner.solve()
                if out is not None:
                    new_rig, rep = out
                    rig_b = new_rig
                    solver.rig_b = new_rig
                    solver.ekf.reset()
                    solver.rig_suspect = False
                    solver._bad_frames = 0
                    print(f"\nphone realigned: moved {rep['moved_cm']:.1f} cm, "
                          f"rotated {rep['rotated_deg']:.1f} deg")
                    print(f"  baseline now {rep['baseline_cm']:.1f} cm, "
                          f"angle {rep['angle_deg']:.1f} deg")
                    print(f"  residual {rep['residual_px']:.2f} px from "
                          f"{rep['inliers']}/{rep['total_points']} inlier points")
                    print(f"  scale x{rep['scale_applied']:.3f} (IQR {rep['scale_iqr']:.3f}), "
                          f"bone residual {rep['bone_residual_mm']:.2f} mm")
                    if rep['bone_residual_mm'] > 4.0:
                        print("  NOTE: bone residual is high -- the rig is still not "
                              "metrically right. More overlap between the two views "
                              "would help most.")
                    save_rig(args.rig, rig_a, rig_b, rep)
                    print(f"  saved to {args.rig}\n")
                else:
                    print(f"realign failed: {realigner.message}")

        health.update(res.ok)
        mode_counts[res.mode] = mode_counts.get(res.mode, 0) + 1

        # Persist the hand model the moment it converges. Leaving this to a
        # keypress meant that a session which failed for other reasons threw
        # the measurement away and started from nothing next time.
        if model.calibrated and not _hand_saved:
            os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
            model.save(CALIB_PATH)
            _hand_saved = True
            print(f"hand model learned and saved to {CALIB_PATH}")
        if np.isfinite(res.reprojection_px):
            reproj.add(res.reprojection_px)
        if np.isfinite(res.bone_residual_mm):
            bone_res.add(res.bone_residual_mm)
        if np.isfinite(res.depth_agreement_mm):
            depth_dis.add(res.depth_agreement_mm)

        if res.ok:
            f = filt(res.points, target)
            if use_pred:
                lead = min(float(np.median(e2e.buf)) / 1000.0, 0.06) if len(e2e.buf) > 10 else 0.03
                f = pred(f, target, lead=lead)
            points = f
            j3.update(f[INDEX_TIP], target)
        else:
            filt.reset(); pred.reset()
        timer.mark("filter")

        # --- draw ------------------------------------------------------------
        panels = []
        for view, rig in ((view_a, rig_a), (view_b, rig_b)):
            if view is None or view.frame is None:
                continue
            canvas = view.frame.image.copy()
            mode = RENDER_MODES[render_mode]
            if mode in ("both", "raw") and view.pixels is not None:
                draw_hand(canvas, view.pixels, thickness=1, radius=2, alpha=0.4)
            if mode in ("both", "processed") and points is not None and rig is not None:
                proj = rig.intrinsics.project(rig.world_to_cam(points))
                draw_hand(canvas, proj, thickness=2, radius=4)
            cv2.putText(canvas, view.name, (12, canvas.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            panels.append(canvas)

        if not panels:
            continue
        h = max(p.shape[0] for p in panels)
        canvas = np.zeros((h, sum(p.shape[1] for p in panels), 3), np.uint8)
        x = 0
        for p in panels:
            canvas[:p.shape[0], x:x + p.shape[1]] = p
            x += p.shape[1]

        # Latency is measured against the NEWEST frame that reached the solver.
        stamps_all = [v.frame_timestamp for v in (view_a, view_b)
                      if v is not None and v.frame_timestamp]
        if stamps_all:
            e2e.add((time.perf_counter() - max(stamps_all)) * 1000.0)

        if solver.rig_suspect and not (realigner and realigner.active):
            cv2.putText(canvas, "RIG CALIBRATION LOOKS WRONG - phone ignored. Press x to realign.",
                        (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 130, 255), 2, cv2.LINE_AA)

        if realigner is not None and realigner.active:
            cv2.putText(canvas, f"REALIGNING  {realigner.progress*100:.0f}%  -- {realigner.message}",
                        (14, canvas.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (90, 255, 140), 2, cv2.LINE_AA)

        if show_hud:
            pos = "--"
            if points is not None:
                p = points[INDEX_TIP] * 100
                pos = f"{p[0]:+5.1f} {p[1]:+5.1f} {p[2]:5.1f} cm"
            fps_a = f"{view_a.source.measured_fps:4.1f}" if view_a else "  --"
            fps_b = f"{view_b.source.measured_fps:4.1f}" if view_b else "  --"
            draw_hud(canvas, [
                ("mode", res.mode),
                ("uncertainty", "--" if not np.isfinite(res.sigma_mm) else f"{res.sigma_mm:5.2f} mm"),
                ("rig", "SUSPECT - press x" if solver.rig_suspect else "ok"),
                ("fps laptop / phone", f"{fps_a} / {fps_b}"),
                ("loop fps", f"{loopfps.median:5.1f}"),
                ("latency", f"{e2e.median:5.1f} ms  p95 {e2e.p95:.0f}"),
                ("frame skew", f"{skew.median:5.1f} ms" if skew.buf else "  n/a"),
                ("depth samples", f"{res.n_depth_samples:3d} / 21" if has_depth else "  n/a"),
                ("solved", f"{health.rate*100:5.1f} %   drops {health.dropouts}"),
                ("reprojection", f"{reproj.median:5.2f} px" if reproj.buf else "  --"),
                ("bone residual", f"{bone_res.median:5.2f} mm" if bone_res.buf else "  --"),
                ("depth disagree", f"{depth_dis.median:5.2f} mm" if depth_dis.buf else "  --"),
                ("hand calib", ("frozen" if model.frozen else "ready") if model.calibrated
                               else f"{model.progress*100:.0f} %"),
                ("index tip", pos),
                ("3D jitter", "hold still" if np.isnan(j3.jitter) else f"{j3.jitter:5.2f} mm"),
            ], origin=(12, 30), width=360)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            render_mode = (render_mode + 1) % len(RENDER_MODES)
        elif key == ord("p"):
            use_pred = not use_pred
        elif key == ord("h"):
            show_hud = not show_hud
        elif key == ord("c"):
            model = HandModel(calibration_frames=90)
            solver.model = model
            print("relearning hand geometry...")
        elif key == ord("x"):
            if realigner is None:
                print("realign needs both cameras and a rig calibration")
            elif realigner.active:
                # Restart rather than cancel: an accidental second press should
                # not silently abandon a collection run.
                realigner.start(); print("realign restarted")
            else:
                realigner.start()
                print("REALIGN: hold your hand where BOTH cameras see it and move it "
                      "around slowly...")
        elif key == ord("f"):
            model.freeze()
            print("hand model frozen (bone residual is now a drift detector)")
        elif key == ord("s") and model.calibrated:
            os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
            model.save(CALIB_PATH)
            print(f"saved hand calibration to {CALIB_PATH}")

    print("\n--- session summary ---")
    print(f"latency        {e2e.median:.1f} ms median, {e2e.p95:.1f} ms p95")
    print(f"solved         {health.rate*100:.1f}%  dropouts {health.dropouts}")
    if skew.buf:
        print(f"frame skew     {skew.median:.1f} ms")
    if reproj.buf:
        print(f"reprojection   {reproj.median:.2f} px")
    if bone_res.buf:
        print(f"bone residual  {bone_res.median:.2f} mm")
    if depth_dis.buf:
        print(f"depth disagree {depth_dis.median:.2f} mm")
    print(f"3D jitter      {j3.jitter:.2f} mm")
    total = sum(mode_counts.values()) or 1
    print("modes used:")
    for m, c in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<18} {c*100.0/total:5.1f} %")

    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

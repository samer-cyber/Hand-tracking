"""Hand-driven mouse control.

Maps your tracked hand onto the screen and, once you enable it, drives the
cursor. Pinch thumb to index to click; hold the pinch to drag.

Calibration teaches it your comfortable reach: you point at each screen corner
in turn, and the region you traced becomes the whole display. That is what
makes it possible to reach every pixel without stretching -- the mapping adapts
to you rather than the other way round.

    python run_mouse.py --laptop 0 --phone 2      # calibrated stereo rig
    python run_mouse.py --phone none              # laptop only (rough, but works)

SAFETY -- read this before enabling control:

  F9        toggle cursor control      (works even without window focus)
  F10       toggle clicking            (works even without window focus)
  ESC hold  emergency stop, kills control and releases buttons
  q         quit (in the preview window)

Cursor control and clicking both start OFF, and you turn them on deliberately.
This is not excessive caution: a tracker that owns the pointer and can only be
stopped by clicking something is genuinely hard to escape from, because
clicking is the thing you have lost control of. Tracking loss also stops
movement immediately rather than leaving the cursor stranded.
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
from handtrack.metrics import Rolling, TrackingHealth
from handtrack.mouse import VK_ESCAPE, VK_F9, VK_F10, MouseController, VirtualScreen, _Edge, key_held
from handtrack.pipeline import HandSolver
from handtrack.runtime import Interrupt
from handtrack.screenmap import ClickAnchor, CursorStabiliser, PinchDetector, ScreenMapper
from handtrack.skeleton import HandModel
from handtrack.sources import describe_spec, open_source
from handtrack.viz import draw_hand, draw_hud

HAND_CALIB = "config/calib/hand_model.npz"
SCREEN_CALIB = "config/calib/screen_map.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laptop", default="0")
    ap.add_argument("--phone", default="none")
    ap.add_argument("--rig", default="config/calib/rig.json")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--recalibrate", action="store_true", help="Redo the screen mapping")
    ap.add_argument("--dead-zone", type=float, default=2.5, help="Cursor dead zone in pixels")
    ap.add_argument("--pinch-on", type=float, default=0.34)
    ap.add_argument("--pinch-off", type=float, default=0.48)
    args = ap.parse_args()

    use_a = str(args.laptop).lower() != "none"
    use_b = str(args.phone).lower() != "none"

    screen = VirtualScreen()
    mouse = MouseController(screen)
    print(f"screen: {screen}")

    src_a = open_source(args.laptop, args.width, args.height, name="laptop") if use_a else None
    src_b = None
    if use_b:
        try:
            src_b = open_source(args.phone, args.width, args.height, name="phone")
        except Exception as exc:
            print(f"phone unavailable ({exc}); continuing without it")
            use_b = False

    has_depth = use_b and str(args.phone).lower() in ("record3d", "phone", "iphone")

    rig_a = rig_b = None
    if use_a and use_b and os.path.exists(args.rig):
        rig_a, rig_b, meta = load_rig(args.rig)
        rms = meta.get("rms", {}).get("stereo", float("nan"))
        quality = f"RMS {rms:.2f} px" if rms == rms else meta.get("method", "calibrated")
        print(f"rig: {meta['angle_deg']:.1f} deg, {quality}")
        adapt_rig_to_sources(rig_a, rig_b, src_a, src_b)
    elif use_b and has_depth:
        rig_b = CameraRig(PinholeCamera.from_fov(src_b.width, src_b.height, args.hfov, "phone"),
                          np.eye(3), np.zeros(3), "phone")
    elif use_a:
        rig_a = CameraRig(PinholeCamera.from_fov(src_a.width, src_a.height, args.hfov, "laptop"),
                          np.eye(3), np.zeros(3), "laptop")
        if use_b:
            print("no rig calibration -- the phone cannot contribute; laptop only.")
            src_b.release(); src_b, use_b = None, False

    det_a = HandDetector(args.model, num_hands=1) if use_a else None
    det_b = HandDetector(args.model, num_hands=1) if use_b else None
    model = HandModel(calibration_frames=90)
    model.load(HAND_CALIB)
    solver = HandSolver(rig_a, rig_b, model)

    filt = OneEuroFilter(1.5, 0.06)
    pred = VelocityPredictor(lead_seconds=0.030, max_step=0.08)
    mapper = ScreenMapper(screen.width, screen.height)
    if not args.recalibrate and mapper.load(SCREEN_CALIB):
        print(f"loaded screen mapping from {SCREEN_CALIB}")
    pinch = PinchDetector(args.pinch_on, args.pinch_off)
    stabiliser = CursorStabiliser(args.dead_zone)
    anchor = ClickAnchor()
    health = TrackingHealth()
    e2e = Rolling(120)

    f9, f10 = _Edge(VK_F9), _Edge(VK_F10)
    calibrating = not mapper.calibrated
    _hand_saved = model.calibrated
    was_pinching = False
    lost_frames = 0
    points = None
    la = lb = -1

    if calibrating:
        print("\n=== screen calibration ===")
        print("Point at each corner of your screen in turn and press SPACE.")
        print("Hold your hand where it is comfortable -- that reach becomes the")
        print("whole screen, so do not stretch. 'u' undoes, 'q' quits.\n")

    interrupt = Interrupt().install()
    done = {"shut": False}

    def shutdown():
        # Idempotent, and registered so it runs however the loop ends. A held
        # mouse button or a locked camera left behind is worse than any crash.
        if done["shut"]:
            return
        done["shut"] = True
        try:
            mouse.set_enabled(False)
            mouse.release_all()
        except Exception:
            pass
        for s_ in (src_a, src_b):
            if s_ is not None:
                try: s_.release()
                except Exception: pass
        for d_ in (det_a, det_b):
            if d_ is not None:
                try: d_.close()
                except Exception: pass
        cv2.destroyAllWindows()

    atexit.register(shutdown)

    win = "Hand mouse"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 620)

    while True:
        if interrupt:
            break
        fa = src_a.read() if src_a else None
        fb = src_b.read() if src_b else None
        if (fa is None or fa.index == la) and (fb is None or fb.index == lb):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

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

        depth = None
        if has_depth and px_b is not None and fb is not None:
            from handtrack.sources.record3d_source import sample_depth
            depth = sample_depth(fb, px_b)

        ts = max([f.timestamp for f in (fa, fb) if f is not None])
        res = solver.solve(px_a, px_b, depth, ts, world_a=world_a, world_b=world_b)
        health.update(res.ok)

        if res.ok:
            f = filt(res.points, ts)
            lead = min(float(np.median(e2e.buf)) / 1000.0, 0.06) if len(e2e.buf) > 10 else 0.03
            points = pred(f, ts, lead=lead)
            lost_frames = 0
        else:
            lost_frames += 1
            filt.reset(); pred.reset()
            # Never leave the cursor being driven by a stale pose.
            if lost_frames > 3:
                points = None
                stabiliser.reset()
                if mouse.left_down:
                    mouse.release_left()

        # --- global safety keys, checked every frame -------------------------
        if key_held(VK_ESCAPE):
            if mouse.enabled or mouse.clicks_enabled:
                print("ESC -- emergency stop")
            mouse.set_enabled(False)
            mouse.clicks_enabled = False
        if f9.pressed():
            if not mapper.calibrated:
                print("calibrate the screen mapping first")
            else:
                mouse.set_enabled(not mouse.enabled)
                print(f"cursor control {'ON' if mouse.enabled else 'off'}")
        if f10.pressed():
            mouse.clicks_enabled = not mouse.clicks_enabled
            if not mouse.clicks_enabled:
                mouse.release_left()
            print(f"clicking {'ON' if mouse.clicks_enabled else 'off'}")

        # --- drive the pointer ----------------------------------------------
        cursor = None
        if points is not None and mapper.calibrated:
            mapped = mapper.to_screen(points[INDEX_TIP])
            if mapped is not None:
                now = time.perf_counter()
                is_pinch = pinch.update(points)
                started = is_pinch and not was_pinching
                mapped = anchor.apply(mapped, started, is_pinch, now)
                cursor = stabiliser(*mapped)
                mouse.move_to(*cursor)

                if started:
                    mouse.press_left()
                elif was_pinching and not is_pinch:
                    mouse.release_left()
                was_pinching = is_pinch
        elif points is not None:
            pinch.update(points)

        e2e.add((time.perf_counter() - ts) * 1000.0)

        # --- draw -------------------------------------------------------------
        base = (fa if fa is not None else fb)
        canvas = base.image.copy()
        canvas = cv2.flip(canvas, 1)
        h_img, w_img = canvas.shape[:2]
        src_px = px_a if px_a is not None else px_b
        if src_px is not None:
            mirrored = np.column_stack([w_img - src_px[:, 0], src_px[:, 1]])
            draw_hand(canvas, mirrored, thickness=2, radius=3)

        if calibrating:
            idx = mapper.progress
            if idx < 4:
                name = ScreenMapper.CORNER_NAMES[idx]
                cv2.putText(canvas, f"Point at the {name.upper()} corner, press SPACE",
                            (20, h_img - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 255, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"{idx}/4 recorded    'u' undo    'q' quit",
                            (20, h_img - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            state = "CONTROL ON" if mouse.enabled else "control off (F9)"
            colour = (80, 130, 255) if mouse.enabled else (170, 170, 170)
            cv2.putText(canvas, state, (20, h_img - 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour, 2, cv2.LINE_AA)

        draw_hud(canvas, [
            ("mode", res.mode),
            ("latency", f"{e2e.median:5.1f} ms"),
            ("tracked", f"{health.rate*100:5.1f} %"),
            ("screen map", "ready" if mapper.calibrated else f"{mapper.progress}/4"),
            ("cursor", f"{cursor[0]:.0f}, {cursor[1]:.0f}" if cursor else "--"),
            ("pinch ratio", "--" if np.isnan(pinch.ratio) else f"{pinch.ratio:5.2f}"),
            ("pinching", "YES" if pinch.pinching else "no"),
            ("control (F9)", "ON" if mouse.enabled else "off"),
            ("clicks (F10)", "ON" if mouse.clicks_enabled else "off"),
        ], origin=(12, 28), width=330)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and calibrating and points is not None:
            n = mapper.add_sample(points[INDEX_TIP])
            print(f"  recorded {ScreenMapper.CORNER_NAMES[n-1]} ({n}/4)")
            if n >= 4:
                if mapper.finish():
                    mapper.save(SCREEN_CALIB)
                    calibrating = False
                    print(f"\nscreen mapping saved to {SCREEN_CALIB}")
                    print("press F9 to enable cursor control, F10 for clicking\n")
                else:
                    print("  fit failed -- the four points were too close together; starting over")
                    mapper.reset()
        elif key == ord("u") and calibrating:
            print(f"  undone ({mapper.undo_sample()}/4)")
        elif key == ord("c"):
            mapper.reset(); calibrating = True
            mouse.set_enabled(False)
            print("recalibrating screen mapping")

    mouse.set_enabled(False)
    mouse.release_all()
    for s in (src_a, src_b):
        if s:
            s.release()
    for d in (det_a, det_b):
        if d:
            d.close()
    cv2.destroyAllWindows()
    print("\nstopped; cursor control released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

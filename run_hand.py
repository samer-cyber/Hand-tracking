"""Hand tracking and screen control, driven by the phone.

The phone alone is a complete tracker: it measures 21 landmarks, a depth at
each, and reports its own intrinsics. That combination gives metric 3D with no
camera calibration at all -- a pixel fixes the direction, depth fixes the
distance. The laptop camera is optional and only widens coverage.

    python run_hand.py                       # phone only, no calibration
    python run_hand.py --laptop 0            # add the laptop camera

Calibration is of the SCREEN, not the cameras: point at each corner twice, from
two positions at least 30 cm apart. That builds a 3D model of the display, which
every pointing mode then shares.

Keys
    SPACE   record a corner (TOUCH it with your fingertip)
    r       rotate the phone image (fixes a sideways or mirrored hand)
    m       cycle pointing mode
    F9      cursor control on/off      (works without window focus)
    F10     clicking on/off            (works without window focus)
    ESC     emergency stop             (works without window focus)
    c       redo screen calibration
    p       prediction on/off
    q       quit
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
from handtrack.ekf import HandEKF
from handtrack.filters import OneEuroFilter, VelocityPredictor
from handtrack.fusion import CameraRig
from handtrack.metrics import JitterMeter, Rolling, TrackingHealth
from handtrack.mouse import VK_ESCAPE, VK_F9, VK_F10, MouseController, VirtualScreen, _Edge, key_held
from handtrack.pointing import MODE_HELP, Pointer
from handtrack.runtime import Interrupt
from handtrack.screen3d import CORNER_NAMES, ScreenGeometry, TouchCollector
from handtrack.screenmap import ClickAnchor, CursorStabiliser, PinchDetector
from handtrack.skeleton import HandModel
from handtrack.detect import MIDDLE_MCP, WRIST
from handtrack.sources.handbridge_source import (DEFAULT_PORT, HandBridgeSource,
                                                 rotate_landmarks, turns_for_orientation)

HAND_CALIB = "config/calib/hand_model.npz"
SCREEN_CALIB = "config/calib/screen3d.json"


def back_project(rig: CameraRig, pixels: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Landmarks in 3D: pixel gives the direction, depth gives the distance."""
    rays = rig.undistorted_rays(pixels)
    forward = rig.R[2]
    cos = rays @ forward
    out = np.full((21, 3), np.nan)
    ok = np.isfinite(depth) & (np.abs(cos) > 1e-6) & np.all(np.isfinite(rays), axis=1)
    out[ok] = rig.centre + rays[ok] * (depth[ok] / cos[ok])[:, None]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--laptop", default="none", help="camera index, or 'none'")
    ap.add_argument("--model", default="models/hand_landmarker.task")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--turns", type=int, default=None,
                    help="quarter turns to rotate the phone image; omit to use "
                         "the orientation the phone reports")
    ap.add_argument("--min-cutoff", type=float, default=1.2)
    ap.add_argument("--beta", type=float, default=0.05)
    args = ap.parse_args()

    screen = VirtualScreen()
    mouse = MouseController(screen)
    print(f"screen: {screen}")

    print(f"\nwaiting for HandBridge on UDP {args.port}...")
    phone = HandBridgeSource(port=args.port)
    try:
        phone.start(timeout=90.0,
                    on_wait=lambda n, left: print(f"  {left:.0f}s left", flush=True))
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1
    first = phone.read()
    print(f"connected: {first.width}x{first.height}, "
          f"fx={first.intrinsics[0,0]:.0f} (measured by the phone)")

    # The phone defines the world frame, so no camera calibration is needed.
    turns = args.turns if args.turns is not None else turns_for_orientation(first.orientation)
    print(f"phone orientation {first.orientation} -> rotating {turns} quarter turn(s); "
          "press 'r' to change")

    def build_rig(t_):
        px, K, w, h = rotate_landmarks(first.pixels, first.intrinsics,
                                       first.width, first.height, t_)
        return CameraRig(PinholeCamera(K, np.zeros(5), w, h, "phone"),
                         np.eye(3), np.zeros(3), "phone")

    rig = build_rig(turns)

    # Optional laptop camera, for coverage when the hand leaves the phone's view.
    laptop_src = laptop_det = None
    if str(args.laptop).lower() != "none":
        try:
            from handtrack.sources import open_source
            laptop_src = open_source(args.laptop, 1280, 720, name="laptop")
            laptop_det = HandDetector(args.model, num_hands=1)
            print(f"laptop camera {laptop_src.width}x{laptop_src.height} "
                  "(needs a rig calibration to contribute; ignored otherwise)")
        except Exception as exc:
            print(f"laptop camera unavailable: {exc}")
            laptop_src = None

    model = HandModel(calibration_frames=90)
    model.load(HAND_CALIB)
    ekf = HandEKF()
    filt = OneEuroFilter(args.min_cutoff, args.beta)
    pred = VelocityPredictor(lead_seconds=0.02, max_step=0.08)

    geo = ScreenGeometry(screen.width, screen.height)
    collector = TouchCollector()
    calibrating = args.recalibrate or not geo.load(SCREEN_CALIB)
    if not calibrating:
        diag = geo.diagonal_inches()
        print(f"loaded screen model: {diag:.1f} inch diagonal "
              f"({geo.size_m[0]*100:.0f} x {geo.size_m[1]*100:.0f} cm)")
        # A saved calibration is trusted silently otherwise, and a wrong one
        # makes every pointing mode unusable in a way that looks like bad
        # tracking rather than bad calibration.
        if not (10.0 < diag < 22.0):
            print("  !! That is not a plausible laptop screen. Pointing will be")
            print("     unusable until this is redone -- press 'c', or run with")
            print("     --recalibrate.")

    pointer = Pointer(geo, (screen.width, screen.height))
    pinch = PinchDetector()
    stabiliser = CursorStabiliser(dead_zone_px=2.0)
    anchor = ClickAnchor()
    health = TrackingHealth()
    e2e, loop_hz = Rolling(120), Rolling(120)
    jitter = JitterMeter(still_threshold_mps=0.05, scale=1000.0)

    f9, f10 = _Edge(VK_F9), _Edge(VK_F10)
    interrupt = Interrupt().install()
    region_samples: list[np.ndarray] = []
    points = None
    was_pinching = False
    last_index = -1
    last_t = time.perf_counter()
    lost = 0
    done = {"shut": False}

    def shutdown():
        if done["shut"]:
            return
        done["shut"] = True
        try:
            mouse.set_enabled(False)
            mouse.release_all()
        except Exception:
            pass
        phone.release()
        if laptop_src is not None:
            try: laptop_src.release()
            except Exception: pass
        if laptop_det is not None:
            try: laptop_det.close()
            except Exception: pass
        cv2.destroyAllWindows()

    atexit.register(shutdown)

    if calibrating:
        print("\n=== screen calibration ===")
        print("TOUCH each corner of your screen with your index fingertip,")
        print("then press SPACE. Four corners, once each.")
        print("Touch it, do not point at it: the fingertip IS the corner, so")
        print("accuracy comes from tracking rather than from how well you aim.\n")

    win = "Hand tracking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 620)
    canvas_size = (620, 900)

    while not interrupt:
        frame = phone.read()
        if frame is None or frame.index == last_index:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        last_index = frame.index

        now = time.perf_counter()
        dt = now - last_t
        loop_hz.add(1.0 / max(dt, 1e-4))
        last_t = now

        # --- build 3D ------------------------------------------------------
        px_rot, K_rot, w_rot, h_rot = rotate_landmarks(
            frame.pixels, frame.intrinsics, frame.width, frame.height, turns)
        has_hand = frame.has_hand and np.all(np.isfinite(px_rot[INDEX_TIP]))
        n_depth = int(np.isfinite(frame.depth).sum())
        ekf.predict(frame.timestamp)

        if has_hand:
            if n_depth >= 6:
                seeded = back_project(rig, px_rot, frame.depth)
                ekf.seed(seeded, frame.timestamp, only_missing=ekf.initialised)
            if ekf.initialised:
                ekf.update_camera(rig, px_rot, sigma_px=2.5)
                if n_depth >= 6:
                    ekf.update_depth(rig, frame.depth, sigma_m=0.006)
                raw = ekf.positions()
                if np.all(np.isfinite(raw)):
                    model.observe(raw - raw.mean(0))
                    if model.calibrated:
                        from handtrack.fusion import refine_with_skeleton
                        ekf.apply_constraint(
                            refine_with_skeleton(raw, model.bone_lengths, [], []),
                            strength=0.35)
                    smoothed = filt(ekf.positions(), frame.timestamp)
                    points = pred(smoothed, frame.timestamp, lead=0.02)
                    lost = 0
                    jitter.update(points[INDEX_TIP], frame.timestamp)
        else:
            lost += 1
            if lost > 4:
                points = None
                ekf.reset()
                filt.reset(); pred.reset(); stabiliser.reset()
                if mouse.left_down:
                    mouse.release_left()

        # Palm length is a fact we already know (about 9 cm). If the
        # reconstruction disagrees, the 3D is wrong and every downstream
        # symptom -- shake, failed pinches, unusable pointing -- follows from it.
        dist_cm = palm_cm = None
        if points is not None and np.all(np.isfinite(points[[WRIST, MIDDLE_MCP]])):
            dist_cm = float(np.nanmedian(points[:, 2]) * 100)
            palm_cm = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]) * 100)

        health.update(points is not None)
        e2e.add((time.perf_counter() - frame.timestamp) * 1000.0)

        # --- safety keys ---------------------------------------------------
        if key_held(VK_ESCAPE):
            if mouse.enabled or mouse.clicks_enabled:
                print("ESC -- emergency stop")
            mouse.set_enabled(False)
            mouse.clicks_enabled = False
        if f9.pressed():
            if not geo.calibrated:
                print("calibrate the screen first")
            else:
                mouse.set_enabled(not mouse.enabled)
                print(f"cursor control {'ON' if mouse.enabled else 'off'}")
        if f10.pressed():
            mouse.clicks_enabled = not mouse.clicks_enabled
            if not mouse.clicks_enabled:
                mouse.release_left()
            print(f"clicking {'ON' if mouse.clicks_enabled else 'off'}")

        # --- pointing ------------------------------------------------------
        cursor = None
        if points is not None and geo.calibrated and not calibrating:
            region_samples.append(points[INDEX_TIP].copy())
            if len(region_samples) > 900:
                region_samples.pop(0)
            # Refit continuously rather than freezing after the first couple of
            # seconds. A region fixed from a barely-moving hand is tiny, so every
            # later position lands outside it and clamps to a screen edge.
            if len(region_samples) > 60 and len(region_samples) % 30 == 0:
                pointer.region.fit(np.array(region_samples))

            target = pointer.update(points, dt)
            if target is not None:
                is_pinch = pinch.update(points)
                started = is_pinch and not was_pinching
                target = anchor.apply(target, started, is_pinch, now)
                cursor = stabiliser(*target)
                mouse.move_to(*cursor)
                if started:
                    mouse.press_left()
                elif was_pinching and not is_pinch:
                    mouse.release_left()
                was_pinching = is_pinch
        elif points is not None:
            pinch.update(points)

        # --- draw ----------------------------------------------------------
        canvas = np.zeros((*canvas_size, 3), np.uint8)
        if has_hand:
            # The phone sends no image, so draw the landmarks in their own
            # frame, scaled to fit the preview.
            s = min(canvas_size[1] / frame.width, canvas_size[0] / frame.height)
            pts = frame.pixels * s
            from handtrack.viz import draw_hand
            draw_hand(canvas, pts, thickness=2, radius=3)

        if calibrating:
            c = collector.next_target()
            if c is not None:
                cv2.putText(canvas, f"TOUCH the {CORNER_NAMES[c].upper()} corner",
                            (20, canvas_size[0] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (90, 255, 140), 2, cv2.LINE_AA)
                cv2.putText(canvas, "fingertip ON the corner, then SPACE",
                            (20, canvas_size[0] - 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{sum(collector.has(i) for i in range(4))}/4 corners",
                            (20, canvas_size[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (150, 150, 150), 1, cv2.LINE_AA)
        else:
            state = "CONTROL ON" if mouse.enabled else "control off (F9)"
            cv2.putText(canvas, state, (20, canvas_size[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 130, 255) if mouse.enabled else (150, 150, 150), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{pointer.mode}: {MODE_HELP[pointer.mode]}",
                        (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 255, 140), 1, cv2.LINE_AA)

        from handtrack.viz import draw_hud
        draw_hud(canvas, [
            ("phone", f"{phone.measured_fps:5.1f} fps"),
            ("loop", f"{loop_hz.median:5.1f} Hz"),
            ("latency", f"{e2e.median:5.1f} ms"),
            ("transit", f"{phone.transit_ms(frame):5.1f} ms"),
            ("depth", f"{n_depth:2d}/21"),
            ("hand distance", f"{dist_cm:5.1f} cm" if dist_cm else "  --"),
            ("palm length", f"{palm_cm:5.1f} cm" if palm_cm else "  --"),
            ("rotation (r)", f"{turns} turn(s)"),
            ("tracked", f"{health.rate*100:5.1f} %"),
            ("hand model", "ready" if model.calibrated else f"{model.progress*100:.0f} %"),
            ("screen", f"{geo.diagonal_inches():.1f} in" if geo.calibrated else "not set"),
            ("mode (m)", pointer.mode),
            ("cursor", f"{cursor[0]:.0f},{cursor[1]:.0f}" if cursor else "--"),
            ("pinch ratio", "--" if np.isnan(pinch.ratio) else
             f"{pinch.ratio:5.2f} {'PINCH' if pinch.pinching else ''}"),
            ("3D jitter", "hold still" if np.isnan(jitter.jitter) else f"{jitter.jitter:5.2f} mm"),
            ("control F9", "ON" if mouse.enabled else "off"),
            ("clicks F10", "ON" if mouse.clicks_enabled else "off"),
        ], origin=(12, 56), width=330)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            turns = (turns + 1) % 4
            rig = build_rig(turns)
            ekf.reset(); filt.reset(); pred.reset()
            print(f"rotation -> {turns} quarter turn(s)")
        elif key == ord("m"):
            print(f"mode -> {pointer.cycle()}")
        elif key == ord("p"):
            pred.lead_seconds = 0.0 if pred.lead_seconds > 0 else 0.02
            print(f"prediction {'off' if pred.lead_seconds == 0 else 'on'}")
        elif key == ord("c"):
            collector.reset(); calibrating = True
            mouse.set_enabled(False)
            print("recalibrating the screen")
        elif key == ord(" ") and calibrating and points is not None:
            c = collector.next_target()
            if c is not None:
                tip = points[INDEX_TIP]
                if not np.all(np.isfinite(tip)):
                    print("  cannot see your fingertip")
                else:
                    collector.record(c, tip)
                    print(f"  {CORNER_NAMES[c]} touched at "
                          f"({tip[0]*100:6.1f},{tip[1]*100:6.1f},{tip[2]*100:6.1f}) cm")
                    if collector.complete:
                        corners = collector.solve()
                        if corners is not None and geo.set_corners(corners):
                            diag = geo.diagonal_inches()
                            print(f"\nscreen model: {geo.size_m[0]*100:.1f} x "
                                  f"{geo.size_m[1]*100:.1f} cm, {diag:.1f} inch diagonal")
                            if 10.0 < diag < 22.0:
                                geo.save(SCREEN_CALIB)
                                calibrating = False
                                pointer.estimate_eye(corners)
                                print("  saved. F9 for the cursor, m to change mode\n")
                            else:
                                # Refusing to save a bad model matters: a wrong
                                # screen makes the ray modes unusable in a way
                                # that reads as bad tracking, not bad geometry.
                                print("  Not a plausible laptop screen, so NOT saved.")
                                print("  Check the hand looks upright in the preview")
                                print("  (press r to rotate), then touch again.")
                                collector.reset()
                        else:
                            print("  calibration failed -- try again")
                            collector.reset()

    print("\n--- session ---")
    print(f"phone       {phone.measured_fps:.1f} fps, {phone.packets} packets, "
          f"{phone.dropped} dropped")
    print(f"latency     {e2e.median:.1f} ms median, {e2e.p95:.1f} ms p95")
    print(f"tracked     {health.rate*100:.1f} %")
    print(f"3D jitter   {jitter.jitter:.2f} mm")
    if model.calibrated:
        os.makedirs(os.path.dirname(HAND_CALIB), exist_ok=True)
        model.save(HAND_CALIB)
    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

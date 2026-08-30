"""Diagnose the iPhone connection, one layer at a time.

The phone path has several independent pieces that must all be in place, and
when it fails the underlying error is usually an unhelpful socket message. This
checks each layer in order and says exactly which one is broken and what to do
about it.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD, WARN = "  [ok]  ", " [FAIL] ", " [warn] "


def main() -> int:
    print("Checking the iPhone streaming path\n" + "=" * 52)

    # 1. the Python package
    try:
        from record3d import Record3DStream
        print(OK + "record3d package is installed")
    except ImportError:
        print(BAD + "record3d is not installed")
        print("        fix:  pip install record3d")
        return 1

    # 2. Apple's USB device service
    try:
        devices = Record3DStream.get_connected_devices()
    except Exception as exc:
        print(BAD + f"could not query devices: {exc}")
        return 1

    if not devices:
        print(BAD + "no iOS device found")
        print()
        print("        Work through these in order:")
        print("        1. Install the 'Apple Devices' app from the Microsoft Store")
        print("           (or iTunes). Without it Windows has no usbmuxd service and")
        print("           the phone is invisible. A 'usbmuxd ... error opening socket'")
        print("           message above means exactly this.")
        print("        2. Connect the iPhone by cable, unlock it, and tap 'Trust' on")
        print("           the 'Trust This Computer?' prompt.")
        print("        3. Confirm the Apple Devices app can see the phone.")
        print("        4. Open Record3D on the phone, go to Settings, and turn on")
        print("           'USB Streaming mode'.")
        return 1

    print(OK + f"{len(devices)} iOS device(s) visible over USB")
    for d in devices:
        print(f"        product id {d.product_id}  udid {d.udid}")

    # 3. the stream itself
    from handtrack.sources.record3d_source import Record3DSource, sample_depth

    print("\n" + "-" * 52)
    print("Now set the phone up. In the Record3D app:")
    print("  1. Settings tab -> turn ON 'USB Streaming mode'")
    print("     (missing or greyed out? the USB-streaming in-app purchase is")
    print("      not bought -- the free version cannot stream live)")
    print("  2. Switch to the FRONT (TrueDepth) camera")
    print("  3. Press the RECORD button -- the phone only listens while recording")
    print("  4. Keep Record3D in the foreground and the screen unlocked")
    print("-" * 52)
    print("\nWaiting up to 60 s for the stream. Do the above now...")

    def progress(attempt: int, remaining: float) -> None:
        print(f"  ...still waiting ({remaining:.0f} s left, attempt {attempt})", flush=True)

    try:
        src = Record3DSource().start(timeout=60.0, on_wait=progress)
    except Exception as exc:
        print("\n" + BAD + "the stream never started")
        print(f"        {exc}")
        return 1
    print("\n" + OK + "stream is live")

    dt = src.device_type
    if dt == 0:
        print(OK + "device type 0 = TrueDepth (front camera) -- correct for this project")
    elif dt is not None:
        print(WARN + f"device type {dt}; expected 0 (TrueDepth front camera)")
        print("        In Record3D, switch to the FRONT camera. The rear camera on a")
        print("        non-Pro iPhone has no depth sensor at all.")

    # 4. measure what is actually arriving
    print("\nMeasuring the stream for 5 seconds...")
    time.sleep(5.0)
    f = src.read()
    if f is None:
        print(BAD + "no frames received")
        src.release()
        return 1

    print(f"        colour   {f.rgb.shape[1]}x{f.rgb.shape[0]}")
    print(f"        depth    {f.depth.shape[1]}x{f.depth.shape[0]}")
    print(f"        rate     {src.measured_fps:.1f} fps")

    valid = np.isfinite(f.depth) & (f.depth > 0.05) & (f.depth < 2.5)
    pct = 100.0 * valid.mean()
    print(f"        depth coverage {pct:.1f}% of pixels in the 5 cm - 2.5 m range")
    if valid.any():
        d = f.depth[valid]
        print(f"        depth range    {d.min():.2f} - {d.max():.2f} m "
              f"(median {np.median(d):.2f} m)")
    if f.confidence is not None:
        conf = f.confidence
        print(f"        confidence     {100.0*(conf >= 2).mean():.1f}% of pixels at the highest level")
    else:
        print(WARN + "no confidence channel; depth will be used unfiltered")

    K = f.intrinsics
    if np.allclose(K, np.eye(3)):
        print(WARN + "no intrinsics reported; the assumed FOV will be used instead")
    else:
        print(f"        intrinsics     fx {K[0,0]:.1f}  fy {K[1,1]:.1f}  "
              f"cx {K[0,2]:.1f}  cy {K[1,2]:.1f}")

    if src.measured_fps < 15:
        print(WARN + "stream rate is low; check the cable and close other phone apps")
    if pct < 10:
        print(WARN + "very little valid depth. Point the phone at your hands at ~40-60 cm;")
        print("        TrueDepth returns nothing beyond about a metre.")

    print("\nAll good. Next:")
    print("  python run_fusion.py --laptop none --phone record3d")
    print("  (no calibration needed -- depth alone gives metric 3D)")
    src.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

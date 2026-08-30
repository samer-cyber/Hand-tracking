"""Watch what the phone is actually sending.

Run this before anything else once HandBridge connects. It answers, in order,
the questions that matter: are packets arriving at all, is a hand being found,
is depth coming back, and how long the data took to get here.

    python tools/listen_phone.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.runtime import Interrupt
from handtrack.sources.handbridge_source import DEFAULT_PORT, HandBridgeSource
from tools.whats_my_ip import lan_ip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    args = ap.parse_args()

    print(f"Listening on UDP {args.port}")
    print(f"In the HandBridge app, set PC address to {lan_ip()} and port {args.port},")
    print("then press Start.\n")

    src = HandBridgeSource(port=args.port)
    try:
        src.start(timeout=60.0,
                  on_wait=lambda n, left: print(f"  waiting... {left:.0f}s left", flush=True))
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1

    print("packets arriving\n")
    interrupt = Interrupt().install()

    transit, rates = [], []
    # Depth is only meaningful on frames where a hand was actually found.
    # Averaging over every frame buries the real number under the idle ones and
    # produces advice ("move closer") that has nothing to do with the problem.
    coverage_with_hand: list[int] = []
    hand_frames = 0
    total_frames = 0
    last_index = -1
    started = time.time()
    printed = 0

    while not interrupt:
        if args.seconds and time.time() - started > args.seconds:
            break
        frame = src.read()
        if frame is None or frame.index == last_index:
            time.sleep(0.005)
            continue
        last_index = frame.index

        t = src.transit_ms(frame)
        if np.isfinite(t):
            transit.append(t)
        rates.append(src.measured_fps)
        total_frames += 1
        if frame.has_hand:
            hand_frames += 1
            coverage_with_hand.append(int(np.isfinite(frame.depth).sum()))

        printed += 1
        if printed % 15 == 0:
            depth_now = int(np.isfinite(frame.depth).sum())
            tip = frame.pixels[8]
            print(f"  seq {frame.index:6d}  {src.measured_fps:5.1f} fps  "
                  f"hand={'yes' if frame.has_hand else 'no ':>3}  "
                  f"depth {depth_now:2d}/21  "
                  f"index tip ({tip[0]:6.1f},{tip[1]:6.1f}) px  "
                  f"transit {t:6.1f} ms")

    print("\n--- summary ---")
    print(f"packets      {src.packets}   malformed {src.malformed}   dropped {src.dropped}")
    if rates:
        print(f"rate         {np.median(rates):.1f} fps")
    if transit:
        # Phone-side processing plus network, measured against the fastest packet
        # seen. It excludes whatever happens before the frame reaches the phone's
        # own callback, so treat it as a lower bound rather than a total.
        print(f"transit      {np.median(transit):.1f} ms median, "
              f"{np.percentile(transit, 95):.1f} ms p95")
        print("             (phone capture -> arrival here; excludes sensor/exposure)")
    if total_frames:
        print(f"hand seen    {100.0 * hand_frames / total_frames:.0f}% of frames "
              f"({hand_frames}/{total_frames})")
    if coverage_with_hand:
        c = np.array(coverage_with_hand)
        got = c >= 12
        print(f"depth        counted only over frames WITH a hand:")
        print(f"             {100.0 * got.mean():.0f}% of those carried depth, "
              f"{c[got].mean() if got.any() else 0:.0f}/21 landmarks when present")
        # Video and depth can run at different sensor rates, so depth legitimately
        # lands on a fraction of frames. That is a rate difference, not a fault,
        # and it looks alarming on a live readout without this note.
        if 0.30 <= got.mean() <= 0.70:
            print("             depth arrives on roughly half the frames, which is the")
            print("             video/depth rate difference, not a failure. The fusion")
            print("             filter treats depth as optional and handles it.")
        elif got.mean() < 0.15:
            print("             LOW -- move your hand to 30-60 cm from the phone")
    if src.dropped > src.packets * 0.1:
        print("packet loss is high; prefer 5 GHz Wi-Fi, or move closer to the router")

    src.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare the pointing modes on the two things that decide how they feel.

Steadiness: with the hand held still, how much does the cursor wander? This is
what makes small targets hard to hit and what people describe as the tracking
being "jittery", even when the underlying 3D is accurate to a millimetre.

Reach: how much hand movement is needed to cross the whole screen? Too much is
tiring, too little makes precision impossible.

Both are computed from the same measured tracking noise, so the comparison is
about the geometry of each mode rather than about tracking quality.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.detect import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, WRIST
from handtrack.pointing import MODE_ABSOLUTE, MODE_EYE_RAY, MODE_FINGER_RAY, MODE_RELATIVE, Pointer
from handtrack.screen3d import ScreenGeometry

W, H = 0.343, 0.193
SCREEN = (1920, 1080)
TL = np.array([-W / 2, 0.10, 0.0])
CORNERS = np.stack([TL, TL + [W, 0, 0], TL + [W, -H, 0], TL + [0, -H, 0]])


def make_hand(centre: np.ndarray, jitter: float, rng) -> np.ndarray:
    """A hand posed pointing roughly forward, with per-landmark noise."""
    pts = np.full((21, 3), np.nan)
    # The screen is at z = 0 and the hand at z ~ 0.38, so a finger pointing AT
    # the screen must have its tip at smaller z than its knuckle.
    pts[WRIST] = centre + [0.00, -0.08, 0.05]
    pts[INDEX_MCP] = centre + [0.01, -0.02, 0.00]
    pts[MIDDLE_MCP] = centre + [-0.01, -0.02, 0.00]
    pts[INDEX_TIP] = centre + [0.015, 0.02, -0.07]
    good = np.all(np.isfinite(pts), axis=1)
    pts[good] += rng.normal(0, jitter, (good.sum(), 3))
    return pts


def main():
    rng = np.random.default_rng(0)
    geo = ScreenGeometry(*SCREEN)
    geo.set_corners(CORNERS)
    print(f"Screen {geo.diagonal_inches():.1f} inch at 40 cm, "
          f"tracking jitter as measured (~2 mm)\n")

    hand_centre = np.array([0.0, -0.10, 0.38])
    samples = np.array([hand_centre + rng.normal(0, 0.09, 3) for _ in range(60)])

    print(f"{'mode':<14} {'cursor jitter':>14} {'hand travel to cross screen':>30}")
    print("-" * 62)

    for mode in (MODE_EYE_RAY, MODE_FINGER_RAY, MODE_ABSOLUTE, MODE_RELATIVE):
        p = Pointer(geo, SCREEN)
        p.set_mode(mode)
        p.region.fit(samples)
        p.estimate_eye(samples)

        # --- steadiness: hand still, only tracking noise -------------------
        positions = []
        for _ in range(200):
            pts = make_hand(hand_centre, 0.002, rng)
            hit = p.update(pts, 1 / 60)
            if hit is not None:
                positions.append(hit)
        jitter_px = float("nan")
        if len(positions) > 20:
            arr = np.array(positions[20:])          # let relative mode settle
            jitter_px = float(np.sqrt(((arr - arr.mean(0)) ** 2).sum(1).mean()))

        # --- reach: hand movement needed to sweep the full width -----------
        p.relative.reset(centre=True)
        travel = float("nan")
        xs = []
        for dx in np.linspace(-0.18, 0.18, 60):
            pts = make_hand(hand_centre + [dx, 0, 0], 0.0002, rng)
            hit = p.update(pts, 1 / 60)
            if hit is not None:
                xs.append((dx, hit[0]))
        if len(xs) > 5:
            arr = np.array(xs)
            spread = arr[:, 1].max() - arr[:, 1].min()
            if spread > 50:
                # centimetres of hand movement per full screen width
                travel = (arr[:, 0].max() - arr[:, 0].min()) * 100 * (SCREEN[0] / spread)

        jt = "n/a" if not np.isfinite(jitter_px) else f"{jitter_px:6.1f} px"
        tv = "n/a" if not np.isfinite(travel) else f"{travel:5.1f} cm"
        print(f"{mode:<14} {jt:>14} {tv:>30}")

    print("\nSteadiness vs tracking noise, for the two ray modes:")
    print(f"{'jitter':>8} {'eye ray':>12} {'finger ray':>12}")
    print("-" * 34)
    for noise in (0.0005, 0.001, 0.002, 0.004):
        row = []
        for mode in (MODE_EYE_RAY, MODE_FINGER_RAY):
            p = Pointer(geo, SCREEN)
            p.set_mode(mode)
            p.estimate_eye(samples)
            pos = []
            for _ in range(200):
                hit = p.update(make_hand(hand_centre, noise, rng), 1 / 60)
                if hit is not None:
                    pos.append(hit)
            arr = np.array(pos)
            row.append(float(np.sqrt(((arr - arr.mean(0)) ** 2).sum(1).mean()))
                       if len(arr) > 20 else float("nan"))
        print(f"{noise*1000:5.1f} mm {row[0]:9.1f} px {row[1]:9.1f} px")

    print("\nLower is steadier. The ratio between the two columns is the ratio of")
    print("their lever arms -- which is the whole argument for the eye ray.")


if __name__ == "__main__":
    main()

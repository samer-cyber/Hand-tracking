"""Validate the 3D screen model against a known screen.

Sets up a virtual 15.5-inch display at a realistic position, simulates pointing
at each corner from two hand positions with realistic aiming error, and checks
that the recovered geometry matches. Also measures how much cursor error a given
amount of hand-aim error produces, which is the number that decides whether
pointing feels precise or vague.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handtrack.screen3d import CornerCollector, ScreenGeometry, closest_point_to_rays

# A 15.5" 16:9 display: 34.3 cm wide, 19.3 cm tall.
W, H = 0.343, 0.193
SCREEN_PX = (1920, 1080)

# Screen sits upright in front of the user, its top-left 17 cm left of origin.
TL = np.array([-W / 2, 0.10, 0.00])
TR = TL + np.array([W, 0, 0])
BL = TL + np.array([0, -H, 0])
BR = TR + np.array([0, -H, 0])
TRUTH = np.stack([TL, TR, BR, BL])


def point_at(target: np.ndarray, hand: np.ndarray, aim_error_deg: float, rng):
    """A pointing ray from `hand` toward `target`, with angular aiming error."""
    d = target - hand
    d = d / np.linalg.norm(d)
    if aim_error_deg > 0:
        perp = np.cross(d, [0, 0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(d, [0, 1.0, 0])
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(d, perp)
        a = np.radians(rng.normal(0, aim_error_deg, 2))
        d = d + perp * np.tan(a[0]) + perp2 * np.tan(a[1])
        d /= np.linalg.norm(d)
    return hand, d


def main():
    rng = np.random.default_rng(0)

    print("Virtual screen: 15.5 inch, 1920x1080, 34.3 x 19.3 cm\n")

    # --- ray intersection, no noise ---------------------------------------
    o = np.array([0.0, -0.20, 0.45])
    p = closest_point_to_rays(np.array([o, o + [0.25, 0, 0]]),
                              np.array([TL - o, TL - (o + [0.25, 0, 0])]))
    print(f"two-ray corner recovery (exact): error {np.linalg.norm(p - TL)*1000:.4f} mm")

    # --- full calibration with realistic aiming error ----------------------
    print(f"\n{'aim error':>10} {'corner err':>12} {'diagonal':>10} {'cursor err':>12}")
    print("-" * 50)
    for aim_deg in (0.0, 1.0, 2.0, 4.0):
        corner_errs, cursor_errs, diagonals = [], [], []
        for _ in range(30):
            collector = CornerCollector()
            # Two hand positions, ~30 cm apart, as the calibration asks for.
            hands = [np.array([-0.15, -0.25, 0.42]), np.array([0.16, -0.20, 0.38])]
            for c in range(4):
                for p_i, hand in enumerate(hands):
                    o_, d_ = point_at(TRUTH[c], hand, aim_deg, rng)
                    collector.record(c, p_i, o_, d_)
            got = collector.solve()
            if got is None:
                continue
            corner_errs.append(np.linalg.norm(got - TRUTH, axis=1).mean())

            geo = ScreenGeometry(*SCREEN_PX)
            if not geo.set_corners(got):
                continue
            diagonals.append(geo.diagonal_inches())

            # Point at the screen centre and see where the cursor lands.
            centre = TRUTH.mean(0)
            hand = np.array([0.0, -0.22, 0.40])
            o_, d_ = point_at(centre, hand, aim_deg, rng)
            hit = geo.intersect_ray(o_, d_)
            if hit is not None:
                want = (SCREEN_PX[0] / 2, SCREEN_PX[1] / 2)
                cursor_errs.append(np.hypot(hit[0] - want[0], hit[1] - want[1]))

        print(f"{aim_deg:8.1f}d {np.mean(corner_errs)*1000:9.1f} mm "
              f"{np.mean(diagonals):8.1f}\" {np.mean(cursor_errs):9.0f} px")

    print(f"\n(true diagonal is 15.5\"; cursor error is at screen centre)")

    # --- why two passes are required --------------------------------------
    print("\nWhat happens with both samples from nearly the same place:")
    for sep_cm in (30, 10, 3, 1):
        errs = []
        for _ in range(30):
            collector = CornerCollector()
            hands = [np.array([-0.02, -0.22, 0.40]),
                     np.array([-0.02 + sep_cm / 100.0, -0.22, 0.40])]
            for c in range(4):
                for p_i, hand in enumerate(hands):
                    o_, d_ = point_at(TRUTH[c], hand, 1.0, rng)
                    collector.record(c, p_i, o_, d_)
            got = collector.solve()
            if got is not None:
                errs.append(np.linalg.norm(got - TRUTH, axis=1).mean())
        label = f"{sep_cm} cm apart"
        print(f"  {label:<16} corner error {np.mean(errs)*1000:8.1f} mm")

    # --- degenerate input must be refused, not guessed at -------------------
    same = np.array([[0, 0, 0.5], [0, 0, 0.5]], dtype=float)
    par = np.array([[0, 0, 1.0], [0, 0, 1.0]], dtype=float)
    print(f"\nparallel rays rejected: {closest_point_to_rays(same, par) is None}")
    geo = ScreenGeometry(*SCREEN_PX)
    print(f"degenerate corners rejected: {not geo.set_corners(np.zeros((4, 3)))}")


if __name__ == "__main__":
    main()

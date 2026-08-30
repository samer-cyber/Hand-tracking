"""Work out where the phone should physically go.

Rather than guessing an angle, this searches placements and scores each one
against what actually determines tracking quality:

  coverage     fraction of the working volume both cameras can see. This is the
               hard gate -- triangulation needs both views, and the measured
               overlap so far has been 1-23%, which is why so many frames fell
               back to single-camera.
  parallax     the angle between the two viewing rays at each point. Depth
               error scales roughly as 1/sin(parallax), so shallow angles are
               nearly useless and the returns flatten past about 45 degrees.
  error        predicted 3D error, combining the two above with the measured
               landmark noise of the detector.

The working volume is where hands actually are when using a laptop: in front of
the screen, below camera height, roughly 25-55 cm out. There is no point
optimising coverage of places your hands never go.
"""
from __future__ import annotations

import itertools

import numpy as np

# --- laptop geometry ---------------------------------------------------------
# Origin at the laptop camera. X right, Y up, Z forward toward the user.
# The lid leans back, so the camera looks slightly downward toward the user.
LID_TILT_DEG = 15.0          # lid reclined from vertical
LAPTOP_HFOV = 65.0           # typical integrated webcam; measure to refine
PHONE_HFOV = 75.0            # iPhone front camera is wider than a webcam
PHONE_ASPECT = 4.0 / 3.0     # the square sensor can crop either way
LAPTOP_ASPECT = 16.0 / 9.0

# --- working volume ----------------------------------------------------------
# Where a seated user's hands live, in metres, relative to the laptop camera.
VOL_X = (-0.28, 0.28)
VOL_Y = (-0.32, 0.06)
VOL_Z = (0.22, 0.58)
SAMPLES = 14

LANDMARK_NOISE_PX = 1.5


def make_grid():
    xs = np.linspace(*VOL_X, SAMPLES)
    ys = np.linspace(*VOL_Y, SAMPLES)
    zs = np.linspace(*VOL_Z, SAMPLES)
    return np.array(list(itertools.product(xs, ys, zs)))


def look_at(eye: np.ndarray, target: np.ndarray):
    """Rotation taking world points into a camera at `eye` aimed at `target`."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(fwd @ up_hint) > 0.98:
        up_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(up_hint, fwd); right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return np.stack([right, up, fwd])


def visible(points, eye, R, hfov, aspect, near=0.12, far=1.2):
    """Boolean mask of which world points fall inside a camera frustum."""
    cam = (R @ (points - eye).T).T
    z = cam[:, 2]
    ok = (z > near) & (z < far)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_x = np.abs(cam[:, 0] / z)
        tan_y = np.abs(cam[:, 1] / z)
    lim_x = np.tan(np.radians(hfov) / 2.0)
    lim_y = lim_x / aspect
    return ok & (tan_x < lim_x) & (tan_y < lim_y)


def evaluate(phone_pos, points, laptop_eye, laptop_R, aim):
    """Coverage, parallax and predicted error for one phone placement."""
    phone_R = look_at(phone_pos, aim)

    vis_a = visible(points, laptop_eye, laptop_R, LAPTOP_HFOV, LAPTOP_ASPECT)
    vis_b = visible(points, phone_pos, phone_R, PHONE_HFOV, PHONE_ASPECT)
    both = vis_a & vis_b
    coverage = both.mean()
    if coverage < 0.02:
        return coverage, 0.0, np.inf, phone_R

    p = points[both]
    ray_a = p - laptop_eye
    ray_b = p - phone_pos
    ray_a /= np.linalg.norm(ray_a, axis=1, keepdims=True)
    ray_b /= np.linalg.norm(ray_b, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip((ray_a * ray_b).sum(1), -1, 1)))

    # Depth error from triangulation goes as (range * pixel_noise) / (focal *
    # sin(parallax)). Averaged over the volume this ranks placements correctly
    # even though the constant is approximate.
    f_a = (1280 / 2.0) / np.tan(np.radians(LAPTOP_HFOV) / 2.0)
    rng = np.linalg.norm(p - laptop_eye, axis=1)
    sin_par = np.maximum(np.sin(np.radians(parallax)), 1e-3)
    err_mm = (rng * LANDMARK_NOISE_PX / f_a / sin_par) * 1000.0
    return coverage, float(np.median(parallax)), float(np.median(err_mm)), phone_R


def main():
    points = make_grid()
    laptop_eye = np.zeros(3)
    tilt = np.radians(LID_TILT_DEG)
    # Lid reclined -> camera aims slightly downward toward the user.
    laptop_R = look_at(laptop_eye, np.array([0.0, -np.sin(tilt), np.cos(tilt)]))

    vis_a = visible(points, laptop_eye, laptop_R, LAPTOP_HFOV, LAPTOP_ASPECT)
    print(f"working volume: {len(points)} samples, "
          f"{VOL_X[1]*200:.0f} x {(VOL_Y[1]-VOL_Y[0])*100:.0f} x "
          f"{(VOL_Z[1]-VOL_Z[0])*100:.0f} cm in front of the screen")
    print(f"laptop camera alone sees {vis_a.mean()*100:.0f}% of it "
          f"({LAPTOP_HFOV:.0f} deg HFOV, lid reclined {LID_TILT_DEG:.0f} deg)\n")

    aim = np.array([0.0, (VOL_Y[0] + VOL_Y[1]) / 2, (VOL_Z[0] + VOL_Z[1]) / 2])

    results = []
    for dx in np.arange(0.16, 0.46, 0.02):        # right of the laptop camera
        for dy in np.arange(-0.14, 0.11, 0.02):   # above/below it
            for dz in np.arange(-0.06, 0.20, 0.02):  # forward of the screen plane
                pos = np.array([dx, dy, dz])
                cov, par, err, _ = evaluate(pos, points, laptop_eye, laptop_R, aim)
                results.append((cov, par, err, dx, dy, dz))

    # Rank by predicted error, but only among placements that actually see
    # enough of the volume -- accuracy over a tiny region is worthless.
    usable = [r for r in results if r[0] >= 0.55]
    if not usable:
        usable = sorted(results, key=lambda r: -r[0])[:40]
    usable.sort(key=lambda r: r[2])

    print(f"{'cover':>6} {'parallax':>9} {'err':>8}   position from the laptop camera")
    print("-" * 74)
    for cov, par, err, dx, dy, dz in usable[:10]:
        print(f"{cov*100:5.0f}% {par:8.1f}d {err:7.1f}mm   "
              f"right {dx*100:5.1f} cm, {'up' if dy>=0 else 'down'} {abs(dy)*100:4.1f} cm, "
              f"forward {dz*100:5.1f} cm")

    best = usable[0]
    cov, par, err, dx, dy, dz = best
    pos = np.array([dx, dy, dz])
    R = look_at(pos, aim)
    fwd = R[2]
    yaw = np.degrees(np.arctan2(-fwd[0], fwd[2]))
    pitch = np.degrees(np.arcsin(np.clip(fwd[1], -1, 1)))

    print("\n" + "=" * 74)
    print("RECOMMENDED PLACEMENT")
    print("=" * 74)
    print(f"  Position, measured from the laptop camera (centre of the screen bezel):")
    print(f"    {dx*100:.0f} cm to the RIGHT")
    print(f"    {abs(dy)*100:.0f} cm {'ABOVE' if dy >= 0 else 'BELOW'} it")
    print(f"    {dz*100:.0f} cm FORWARD of the screen (toward you)")
    print(f"  Aim: rotate the phone {abs(yaw):.0f} deg to the "
          f"{'LEFT' if yaw > 0 else 'RIGHT'} (toward the screen centre),")
    print(f"       and tilt it {abs(pitch):.0f} deg "
          f"{'UP' if pitch > 0 else 'DOWN'}.")
    print(f"  Phone in PORTRAIT, front camera facing your hands.")
    print(f"\n  Predicted: {cov*100:.0f}% of the working volume seen by both cameras,")
    print(f"             {par:.0f} deg median parallax, ~{err:.1f} mm median 3D error.")
    print(f"\n  For comparison, your measured overlap so far has been 1-23%.")
    print("=" * 74)

    print("\nSensitivity -- how much placement error you can tolerate:")
    for label, delta in [("5 cm nearer the laptop", np.array([-0.05, 0, 0])),
                         ("5 cm further away", np.array([0.05, 0, 0])),
                         ("5 cm higher", np.array([0, 0.05, 0])),
                         ("5 cm further forward", np.array([0, 0, 0.05]))]:
        c, p_, e, _ = evaluate(pos + delta, points, laptop_eye, laptop_R, aim)
        print(f"  {label:<24} coverage {c*100:4.0f}%  parallax {p_:4.1f}d  err {e:6.1f}mm")

    print("\nAim sensitivity (phone rotated away from the ideal target):")
    for off_deg in (0, 5, 10, 20):
        a2 = aim + np.array([np.sin(np.radians(off_deg)) * 0.4, 0, 0])
        c, p_, e, _ = evaluate(pos, points, laptop_eye, laptop_R, a2)
        print(f"  {off_deg:2d} deg off-aim        coverage {c*100:4.0f}%  "
              f"parallax {p_:4.1f}d  err {e:6.1f}mm")


if __name__ == "__main__":
    main()

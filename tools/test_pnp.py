"""Exercise the monocular PnP path with realistic data -- the code path that crashed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from handtrack.cameras import PinholeCamera, MonoPoseSolver
from handtrack.detect import BONES

rng = np.random.default_rng(3)
hand = np.zeros((21, 3))
for c, p in BONES:
    hand[c] = hand[p] + rng.normal(0, 1, 3) * 0.012 + np.array([0.018, 0.016, 0.004])
hand -= hand.mean(0)

intr = PinholeCamera.from_fov(1280, 720, 60.0)
solver = MonoPoseSolver(intr)

ok = 0
for trial, dist in enumerate(np.linspace(0.25, 0.8, 40)):
    ang = trial * 0.15
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    truth = (R @ hand.T).T + np.array([0.02, -0.01, dist])
    px = intr.project(truth) + rng.normal(0, 1.0, (21, 2))
    res = solver.solve(hand, px)
    if res is None:
        print(f"  trial {trial}: solve returned None (dist {dist:.2f} m)")
        continue
    cam_pts, rvec, tvec = res
    err = np.linalg.norm(cam_pts - truth, axis=1).mean()
    depth_err = abs(cam_pts[:, 2].mean() - truth[:, 2].mean())
    ok += 1
    if trial % 10 == 0:
        print(f"  dist {dist:.2f} m -> 3D err {err*1000:5.1f} mm, depth err {depth_err*1000:5.1f} mm")

print(f"\n{ok}/40 solves succeeded, no exceptions raised")

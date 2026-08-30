import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from handtrack.skeleton import HandModel
from handtrack.detect import BONES

rng = np.random.default_rng(0)
base = np.zeros((21, 3))
for c, p in BONES:
    base[c] = base[p] + rng.normal(0, 1, 3) * 0.01 + np.array([0.02, 0.02, 0.002])

m = HandModel(calibration_frames=30)
for _ in range(40):
    m.observe(base + rng.normal(0, 0.0005, (21, 3)))

noisy = base + rng.normal(0, 0.004, (21, 3))
fixed = m.apply(noisy)
eb = np.array([abs(np.linalg.norm(noisy[c]-noisy[p]) - m.bone_lengths[k]) for k, (c, p) in enumerate(BONES)])
ea = np.array([abs(np.linalg.norm(fixed[c]-fixed[p]) - m.bone_lengths[k]) for k, (c, p) in enumerate(BONES)])
print("calibrated:", m.calibrated)
print(f"mean bone-length error  before {eb.mean()*1000:6.2f} mm   after {ea.mean()*1000:6.2f} mm")
print(f"max  bone-length error  before {eb.max()*1000:6.2f} mm   after {ea.max()*1000:6.2f} mm")

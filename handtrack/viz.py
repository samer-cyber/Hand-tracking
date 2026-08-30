"""Drawing helpers for the debug view."""
from __future__ import annotations

import cv2
import numpy as np

from .detect import BONES

FINGER_COLORS = {
    0: (200, 200, 200),   # palm / wrist
    1: (80, 180, 255),    # thumb
    2: (90, 255, 140),    # index
    3: (255, 210, 80),    # middle
    4: (255, 130, 200),   # ring
    5: (150, 150, 255),   # pinky
}


def _finger_of(idx: int) -> int:
    if idx == 0:
        return 0
    return 1 + (idx - 1) // 4


def draw_hand(img: np.ndarray, pts2d: np.ndarray, thickness: int = 2,
              radius: int = 3, alpha: float = 1.0) -> None:
    """Draw a 21-point hand skeleton given pixel coordinates."""
    overlay = img if alpha >= 1.0 else img.copy()
    for c, p in BONES:
        a, b = pts2d[p], pts2d[c]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        cv2.line(overlay, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 FINGER_COLORS[_finger_of(c)], thickness, cv2.LINE_AA)
    for i, p in enumerate(pts2d):
        if not np.all(np.isfinite(p)):
            continue
        cv2.circle(overlay, (int(p[0]), int(p[1])), radius,
                   FINGER_COLORS[_finger_of(i)], -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_hud(img: np.ndarray, lines: list[tuple[str, str]], origin=(12, 24),
             line_height: int = 22, width: int = 330) -> None:
    """Draw a translucent stats panel of label/value pairs."""
    x, y = origin
    h = line_height * len(lines) + 14
    panel = img[max(0, y - 20):y - 20 + h, x - 8:x - 8 + width]
    if panel.size:
        panel[:] = (panel * 0.35).astype(np.uint8)
    for i, (label, value) in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(img, label, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(img, value, (x + 190, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_latency_bar(img: np.ndarray, stages: dict[str, float], origin=(12, 0), width: int = 330):
    """Stacked bar showing where the milliseconds go."""
    x, y = origin
    total = sum(stages.values()) or 1.0
    colors = [(90, 255, 140), (80, 180, 255), (255, 210, 80), (255, 130, 200), (150, 150, 255)]
    cx = x
    for i, (name, ms) in enumerate(stages.items()):
        w = int(width * ms / total)
        cv2.rectangle(img, (cx, y), (cx + w, y + 12), colors[i % len(colors)], -1)
        cx += w
    cv2.putText(img, f"{total:.0f} ms pipeline", (x + width + 10, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

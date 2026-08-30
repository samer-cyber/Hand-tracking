"""MediaPipe Hand Landmarker wrapper.

MediaPipe is used here as a 2D landmark detector, which is the job it is
genuinely good at. Its per-frame 3D output ("world landmarks") is a monocular
guess: the depth axis is inferred from learned hand proportions rather than
measured, so it is scale-ambiguous, noisy along Z, and prone to flipping
between poses that look identical from one viewpoint. Everything downstream of
this module exists to replace that guessed depth with geometry -- triangulation
between two views, and direct depth measurement from the phone.
"""
from __future__ import annotations

from dataclasses import dataclass

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Landmark indices, per MediaPipe's hand topology.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Parent index for every landmark; -1 marks the root. Used for bone-length
# constraints and for drawing.
PARENT = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

BONES = [(i, int(PARENT[i])) for i in range(21) if PARENT[i] >= 0]


@dataclass
class HandObservation:
    """One hand as seen by one camera at one instant."""
    pixels: np.ndarray        # (21,2) landmark positions in image pixels
    normalized: np.ndarray    # (21,3) MediaPipe normalized coords, z is a rough guess
    world: np.ndarray         # (21,3) metric metres, origin at hand centre, monocular
    handedness: str           # "Left" or "Right"
    score: float
    timestamp: float          # capture timestamp of the source frame
    frame_index: int


class HandDetector:
    """Detects hand landmarks on individual frames, tracking timestamps itself."""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        # MediaPipe's VIDEO mode requires strictly increasing integer
        # millisecond timestamps and will raise if one ever goes backwards.
        self._last_ms = -1

    def detect(self, image_bgr: np.ndarray, timestamp: float, frame_index: int = 0) -> list[HandObservation]:
        h, w = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(timestamp * 1000.0)
        if ts_ms <= self._last_ms:
            ts_ms = self._last_ms + 1
        self._last_ms = ts_ms

        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        out: list[HandObservation] = []
        for i, lms in enumerate(result.hand_landmarks):
            norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
            px = np.stack([norm[:, 0] * w, norm[:, 1] * h], axis=1)
            world = np.zeros((21, 3))
            if i < len(result.hand_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=np.float64)
            label, score = "Unknown", 0.0
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
                score = float(result.handedness[i][0].score)
            out.append(HandObservation(
                pixels=px, normalized=norm, world=world,
                handedness=label, score=score,
                timestamp=timestamp, frame_index=frame_index,
            ))
        return out

    def close(self) -> None:
        self.landmarker.close()

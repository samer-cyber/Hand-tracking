"""Print where your index fingertip is, straight from the phone.

Close the HandPilot window first (the phone serves one program at a time),
then from the repository folder:

    python examples/print_landmarks.py
    python examples/print_landmarks.py 192.168.1.20     # over Wi-Fi
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handpilot import HandStream  # noqa: E402

wifi = sys.argv[1] if len(sys.argv) > 1 else ""

with HandStream(wifi=wifi) as hands:
    print("waiting for the phone... (Ctrl+C to stop)")
    for frame in hands:
        if not frame.hand:
            continue
        x, y, confidence = frame.landmark("index_tip")
        print(f"{frame.handedness:>7} hand   index tip x={x:.3f} y={y:.3f}   "
              f"confidence {confidence:.2f}", end="\r")

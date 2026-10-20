"""HandPilot: your iPhone as a high-speed hand tracker for your computer.

Drive the mouse with the app (``python run.py``), or read the landmarks
yourself:

    from handpilot import HandStream

    with HandStream() as hands:
        for frame in hands:
            print(frame.landmark("index_tip"))
"""

__version__ = "1.0.0"

from .protocol import LANDMARK_NAMES, Frame  # noqa: E402
from .stream import HandStream, UdpForwarder  # noqa: E402

__all__ = ["HandStream", "UdpForwarder", "Frame", "LANDMARK_NAMES", "__version__"]

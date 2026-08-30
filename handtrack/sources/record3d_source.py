"""iPhone RGB + TrueDepth streaming over USB, via the Record3D app.

This is the phone half of the rig. Record3D streams three registered things
from the front of the iPhone:

  RGB          the 18 MP square Center Stage sensor, downscaled for streaming
  depth        metric depth in metres from the TrueDepth projector
  confidence   per-pixel 0/1/2 quality flag for the depth

The depth map is registered to the RGB frame, meaning a landmark found at pixel
(x, y) in the colour image can be looked up directly in the depth image (after
accounting for the two having different resolutions). That is what makes the
phone worth so much more than a second webcam: it does not just give a second
viewpoint to triangulate from, it independently *measures* the distance to
every landmark it can see.

The confidence channel matters. TrueDepth is a structured-light sensor -- it
projects a dot pattern and infers depth from its deformation -- so it returns
poor data at glancing angles, on edges where the pattern falls off, and on dark
or shiny surfaces. Reading depth without checking confidence is the main way
this sensor produces confidently wrong numbers, so low-confidence pixels are
discarded here rather than fused.

Requires, on the PC:
  * Apple Devices (Microsoft Store) or iTunes, for the USB device service
  * pip install record3d
and on the phone:
  * the Record3D app, with the USB-streaming unlock purchased
  * Settings -> USB Streaming mode enabled, front TrueDepth camera selected
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    from record3d import Record3DStream
    AVAILABLE = True
except ImportError:                          # pragma: no cover
    Record3DStream = None
    AVAILABLE = False


@dataclass
class RGBDFrame:
    """One synchronised colour + depth observation from the phone."""
    rgb: np.ndarray               # (H,W,3) uint8, BGR to match OpenCV
    depth: np.ndarray             # (h,w) float32 metres, 0 where invalid
    confidence: np.ndarray | None  # (h,w) uint8, 2 = good
    intrinsics: np.ndarray        # 3x3 for the RGB frame
    timestamp: float
    index: int

    @property
    def image(self) -> np.ndarray:
        """Alias so an RGBDFrame can stand in for a plain camera Frame."""
        return self.rgb


class Record3DSource:
    """Threaded receiver for the Record3D USB stream.

    Mirrors CameraSource's contract -- start(), read() returning the newest
    frame, release() -- so the rest of the pipeline does not care whether a
    view came from a webcam or from the phone.
    """

    def __init__(self, device_index: int = 0, name: str = "phone"):
        if not AVAILABLE:
            raise RuntimeError(
                "The record3d package is not installed. Run: pip install record3d")
        self.name = name
        self.device_index = device_index
        self.session: Record3DStream | None = None

        self._lock = threading.Lock()
        self._latest: RGBDFrame | None = None
        self._count = 0
        self._last_arrival: float | None = None
        self._intervals: list[float] = []
        self.device_type: int | None = None

    def start(self, timeout: float = 30.0, on_wait=None) -> "Record3DSource":
        """Connect to the phone, retrying until the app starts streaming.

        Retrying matters here. The iPhone only listens on Record3D's port while
        the app is in the foreground AND actively recording; at any other moment
        usbmuxd reports RESULT_CONNREFUSED (the "Error code=3" message). A
        single connect attempt therefore fails unless the app happens to already
        be streaming, which forces an awkward ordering on the user. Retrying
        lets them start the script first and then press record.
        """
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError(
                "No iPhone found.\n"
                "  1. Is the Apple Devices app (or iTunes) installed? Without it Windows\n"
                "     has no usbmuxd service and the phone is invisible to record3d.\n"
                "  2. Is the phone connected by cable and unlocked, with this PC trusted?")
        if self.device_index >= len(devices):
            raise RuntimeError(f"Device index {self.device_index} but only {len(devices)} connected")

        dev = devices[self.device_index]
        deadline = time.time() + timeout
        attempt = 0

        while time.time() < deadline and self._latest is None:
            attempt += 1
            self.session = Record3DStream()
            self.session.on_new_frame = self._on_new_frame
            self.session.on_stream_stopped = self._on_stopped
            try:
                self.session.connect(dev)
            except Exception:
                pass

            # Give this attempt a moment to deliver a frame before retrying.
            wait_until = min(time.time() + 2.0, deadline)
            while self._latest is None and time.time() < wait_until:
                time.sleep(0.02)

            if self._latest is None and on_wait is not None:
                on_wait(attempt, max(0.0, deadline - time.time()))

        if self._latest is None:
            raise RuntimeError(
                "The phone is connected, but Record3D is not streaming.\n"
                "  usbmuxd reported RESULT_CONNREFUSED, which means nothing on the phone\n"
                "  is listening on Record3D's port. In the app:\n"
                "    1. Settings tab -> turn ON 'USB Streaming mode'.\n"
                "       If that toggle is missing or greyed out, the USB-streaming\n"
                "       in-app purchase has not been bought -- the free version records\n"
                "       but cannot stream live.\n"
                "    2. Switch to the FRONT (TrueDepth) camera.\n"
                "    3. Press the RECORD button. The app only listens while recording.\n"
                "    4. Leave Record3D in the foreground -- iOS suspends the listener\n"
                "       as soon as the app goes to the background or the screen locks.")

        try:
            self.device_type = self.session.get_device_type()
        except Exception:
            self.device_type = None
        return self

    def _on_new_frame(self) -> None:
        try:
            rgb = self.session.get_rgb_frame()
            depth = self.session.get_depth_frame()
        except Exception:
            return
        if rgb is None or depth is None:
            return
        try:
            conf = self.session.get_confidence_frame()
        except Exception:
            conf = None
        try:
            coeffs = self.session.get_intrinsic_mat()
            K = np.array([[coeffs.fx, 0, coeffs.tx],
                          [0, coeffs.fy, coeffs.ty],
                          [0, 0, 1.0]])
        except Exception:
            K = np.eye(3)

        now = time.perf_counter()
        if self._last_arrival is not None:
            self._intervals.append(now - self._last_arrival)
            if len(self._intervals) > 120:
                self._intervals.pop(0)
        self._last_arrival = now

        # Record3D delivers RGB; the rest of the pipeline is OpenCV-native BGR.
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        with self._lock:
            self._count += 1
            self._latest = RGBDFrame(
                rgb=bgr,
                depth=np.asarray(depth, dtype=np.float32),
                confidence=None if conf is None else np.asarray(conf),
                intrinsics=K,
                timestamp=now,
                index=self._count,
            )

    def _on_stopped(self) -> None:
        print(f"[{self.name}] stream stopped by the phone")

    def read(self) -> RGBDFrame | None:
        with self._lock:
            return self._latest

    @property
    def measured_fps(self) -> float:
        if len(self._intervals) < 5:
            return 0.0
        return 1.0 / float(np.median(self._intervals))

    @property
    def width(self) -> int:
        f = self.read()
        return 0 if f is None else f.rgb.shape[1]

    @property
    def height(self) -> int:
        f = self.read()
        return 0 if f is None else f.rgb.shape[0]

    def release(self) -> None:
        if self.session is not None:
            try:
                self.session.disconnect()
            except Exception:
                pass


def sample_depth(frame: RGBDFrame, pixels: np.ndarray, patch: int = 2,
                 min_confidence: int = 2) -> np.ndarray:
    """Read metric depth at each landmark pixel of the phone's colour image.

    Returns (N,) metres, NaN where no trustworthy depth exists.

    Two details make this reliable rather than merely plausible:

    * The depth map and colour image usually differ in resolution, so landmark
      coordinates are rescaled into depth-image space rather than used directly.
    * A single pixel of structured-light depth is noisy and frequently invalid,
      so we take the median over a small patch and ignore zero and
      low-confidence samples. The median specifically, not the mean, because at
      a hand's silhouette a patch straddles the hand and the wall metres
      behind it; a mean would blend those into a plausible-looking depth that
      corresponds to nothing, while the median stays on whichever surface
      occupies most of the patch.
    """
    h_d, w_d = frame.depth.shape[:2]
    h_c, w_c = frame.rgb.shape[:2]
    sx, sy = w_d / float(w_c), h_d / float(h_c)

    out = np.full(len(pixels), np.nan)
    for i, (x, y) in enumerate(np.asarray(pixels, dtype=np.float64)):
        dx, dy = int(round(x * sx)), int(round(y * sy))
        x0, x1 = max(0, dx - patch), min(w_d, dx + patch + 1)
        y0, y1 = max(0, dy - patch), min(h_d, dy + patch + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        block = frame.depth[y0:y1, x0:x1].ravel()
        mask = np.isfinite(block) & (block > 0.05) & (block < 2.5)
        if frame.confidence is not None:
            cblock = frame.confidence[y0:y1, x0:x1].ravel()
            mask &= cblock >= min_confidence
        if mask.sum() >= 3:
            out[i] = float(np.median(block[mask]))
    return out

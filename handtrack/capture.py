"""Threaded camera capture that always yields the newest frame.

The single biggest source of perceived latency in a naive OpenCV loop is that
`cap.read()` returns the OLDEST buffered frame, not the newest. If inference
takes longer than one frame interval, the driver queue backs up and you end up
rendering hands from several frames ago. This module runs the grab loop in its
own thread and keeps only the most recent frame, so inference always operates
on current reality and stale frames are dropped rather than queued.

The second biggest source is auto-exposure: in typical indoor light a webcam
will happily choose a 1/30s shutter, which both caps framerate and smears a
moving hand across dozens of pixels. Motion blur destroys landmark accuracy in
a way no amount of downstream filtering can recover. We force a short manual
exposure and compensate with gain -- noisier, but sharp.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    """A captured image plus the clock reading from just after it arrived."""
    image: np.ndarray
    timestamp: float      # perf_counter seconds, taken immediately after read()
    index: int            # monotonically increasing frame counter


# On Windows the two backends disagree about how to express "manual exposure".
# DirectShow uses the 0.25/0.75 magic constants; Media Foundation uses 0/1.
_AUTO_EXPOSURE_MANUAL = {cv2.CAP_DSHOW: 0.25, cv2.CAP_MSMF: 0.0}
_AUTO_EXPOSURE_AUTO = {cv2.CAP_DSHOW: 0.75, cv2.CAP_MSMF: 1.0}


class CameraOpenError(RuntimeError):
    """A camera could not be opened, or did not respond in time."""


def _open_with_timeout(index: int, backend: int, configure, timeout: float = 8.0):
    """Open and configure a VideoCapture, giving up if the driver blocks.

    This exists because `cv2.VideoCapture` does not merely fail on a virtual
    camera with no active source -- it blocks inside the driver indefinitely,
    in a C call that holds the GIL. That makes the process unkillable from the
    keyboard: Ctrl+C cannot be delivered because the interpreter never gets to
    run between bytecodes. The only defence is to do the open on a throwaway
    daemon thread and abandon it if it does not return.

    The abandoned thread leaks a handle, but it dies with the process, and a
    leaked handle is a far better outcome than a session you cannot exit.

    `configure` runs inside the guarded thread, before the probe read. That
    ordering is not incidental: Media Foundation computes its buffer stride on
    first capture, and changing the frame size afterwards leaves the stride
    disagreeing with the new width. The next read then dies inside the Mat
    constructor with `_step >= minstep`. Configure first, read second.
    """
    result: dict = {}

    def worker() -> None:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                configure(cap)
                # The constructor can return while the first read still blocks,
                # so the probe read belongs inside the guarded thread too.
                cap.read()
            result["cap"] = cap
        except Exception as exc:                      # pragma: no cover
            result["err"] = exc

    thread = threading.Thread(target=worker, daemon=True, name=f"open-cam{index}")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    if "err" in result:
        raise CameraOpenError(f"Camera index {index} failed to open: {result['err']}")
    return result.get("cap")


class CameraSource:
    """A camera read in a background thread, exposing only its latest frame."""

    def __init__(
        self,
        index: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: int = cv2.CAP_MSMF,
        exposure: float | None = -6.0,
        gain: float | None = None,
        fourcc: str = "MJPG",
        name: str = "cam",
    ):
        self.index = index
        self.name = name
        self.backend = backend
        self._requested = (width, height, fps)

        def configure(cap: cv2.VideoCapture) -> None:
            # FOURCC must be set before the frame size or the driver may silently
            # fall back to an uncompressed mode that cannot sustain the framerate.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # A short driver-side buffer is the difference between "responsive"
            # and "swimming through treacle"; not all backends honour it.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = _open_with_timeout(index, backend, configure, timeout=8.0)
        if self.cap is None:
            raise CameraOpenError(
                f"Camera index {index} did not respond within 8 s.\n"
                "  A virtual camera whose source is not running does this: it blocks\n"
                "  inside the driver instead of failing. Iriun, EpocCam and DroidCam\n"
                "  all behave this way when the phone app is not actively streaming.\n"
                "  Start streaming on the phone first, then re-run.\n"
                f"  Use 'python tools/list_cameras.py' to see which indices respond.")
        if not self.cap.isOpened():
            raise CameraOpenError(f"Could not open camera index {index} on backend {backend}")

        if exposure is not None:
            self.set_exposure(exposure, gain)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._count = 0
        self._running = False
        self._thread: threading.Thread | None = None

        # Rolling measure of true inter-arrival time, which is often nowhere
        # near what CAP_PROP_FPS advertises.
        self.read_errors = 0
        self._last_arrival: float | None = None
        self._intervals: list[float] = []

    def set_exposure(self, exposure: float | None, gain: float | None = None) -> None:
        """Force manual exposure. `exposure` is the OpenCV log2-seconds value.

        -6 means roughly 1/64 s, -7 roughly 1/128 s. Lower is sharper and
        darker. Pass None to hand control back to the camera's own metering.
        """
        if exposure is None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_AUTO.get(self.backend, 0.75))
            return
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_MANUAL.get(self.backend, 0.25))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)

    def start(self) -> "CameraSource":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"capture-{self.name}", daemon=True)
        self._thread.start()
        # Block until the first frame lands so callers never see an empty source.
        deadline = time.time() + 5.0
        while self._latest is None and time.time() < deadline:
            time.sleep(0.005)
        if self._latest is None:
            raise RuntimeError(f"Camera '{self.name}' opened but delivered no frames")
        return self

    def _loop(self) -> None:
        errors = 0
        while self._running:
            try:
                ok, img = self.cap.read()
            except cv2.error:
                # A malformed frame should not take the camera down. Drivers
                # occasionally emit one after a mode change or a USB hiccup;
                # the right response is to skip it, not to lose the stream.
                errors += 1
                self.read_errors = errors
                if errors > 200:
                    self._running = False
                time.sleep(0.005)
                continue
            if not ok or img is None:
                time.sleep(0.002)
                continue
            now = time.perf_counter()
            if self._last_arrival is not None:
                self._intervals.append(now - self._last_arrival)
                if len(self._intervals) > 120:
                    self._intervals.pop(0)
            self._last_arrival = now
            with self._lock:
                self._count += 1
                self._latest = Frame(image=img, timestamp=now, index=self._count)

    def read(self) -> Frame | None:
        """Return the newest frame. Never blocks, never returns a stale queue entry."""
        with self._lock:
            return self._latest

    @property
    def measured_fps(self) -> float:
        if len(self._intervals) < 5:
            return 0.0
        return 1.0 / float(np.median(self._intervals))

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "CameraSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.release()

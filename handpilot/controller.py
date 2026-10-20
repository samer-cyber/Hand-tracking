"""The runtime: the link, what each frame drives, and the switch between modes.

Everything runs straight from the link thread as frames arrive, so the path
from a landmark landing on the socket to a cursor moving is a few hundred
microseconds of arithmetic with no queue and no scheduler wakeup in between.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from collections import deque
from ctypes import wintypes

from . import actions, gestures, handmodel, winput
from .filters import Hysteresis, OneEuro2D, Rope, feel
from .gestures import Binding, GestureEngine
from .link import Link
from .stream import UdpForwarder

#: The pointer filters in pixels of a 1080-line screen and rescales afterwards,
#: so a given steadiness feels the same on a laptop panel and a 4K monitor.
REF_HEIGHT = 1080.0


class PointerMode:
    """The hand as a mouse.

    Per frame: tracking point -> calibrated region -> One Euro filter (on the
    phone's capture clock) -> optional lead -> rope dead zone -> screen. Clicks
    come from the thumb-to-index gap, separately, so the point being steered and
    the fingers doing the clicking never fight each other.
    """

    def __init__(self) -> None:
        self.filter = OneEuro2D()
        self.rope = Rope()
        self.left = Hysteresis(0.30, 0.44)
        self.right = Hysteresis(0.30, 0.44)
        self.region = (0.18, 0.15, 0.82, 0.85)
        self.anchor = "knuckles"
        self.flip_x = True
        self.flip_y = False
        self.screen = "primary"
        self.lead_ms = 0.0
        self.click = True
        self.freeze_px = 16.0
        self.lookback_ms = 90.0
        self.right_click = True
        self.scroll_enabled = False
        self.scroll_speed = 1.0
        self.conf_min = 0.35
        self.lost_release_ms = 220.0

        self._trail: deque = deque(maxlen=64)
        self._stillness: deque = deque(maxlen=60)
        self._down: str | None = None
        self._anchor: tuple[float, float] | None = None
        self._frozen = False
        self._last_good = 0.0
        self._last_t: float | None = None
        self._pinch: float | None = None
        self._pinch_middle: float | None = None
        self._scroll_accum = 0.0
        self._scroll_ref = 0.0
        self._scrolling = False
        #: Surfaced to the UI so thresholds can be tuned by watching a number
        #: rather than by guessing.
        self.debug = {"pinch": 1.0, "frozen": False, "down": None, "anchor": None}

    def configure(self, p: dict) -> None:
        self.region = tuple(p.get("region", self.region))
        self.anchor = "pinch" if p.get("anchor") == "pinch" else "knuckles"
        self.flip_x = bool(p.get("flip_x", True))
        self.flip_y = bool(p.get("flip_y", False))
        self.screen = "all" if p.get("screen") == "all" else "primary"
        if p.get("custom_filter"):
            min_cutoff = float(p.get("min_cutoff", 0.7))
            beta = float(p.get("beta", 0.0095))
            deadzone = float(p.get("deadzone_px", 1.8))
        else:
            min_cutoff, beta, deadzone = feel(float(p.get("steadiness", 0.5)))
        self.filter.min_cutoff = min_cutoff
        self.filter.beta = beta
        self.rope.radius = deadzone
        self.lead_ms = float(p.get("lead_ms", 0.0))
        self.click = bool(p.get("click", True))
        self.left.on_below = float(p.get("pinch_on", 0.30))
        self.left.off_above = float(p.get("pinch_off", 0.44))
        self.right.on_below = float(p.get("right_pinch_on", 0.30))
        self.right.off_above = float(p.get("right_pinch_off", 0.44))
        self.freeze_px = float(p.get("click_freeze_px", 16))
        self.lookback_ms = float(p.get("anchor_lookback_ms", 90))
        self.right_click = bool(p.get("right_click", True))
        self.scroll_enabled = bool(p.get("scroll_enabled", False))
        self.scroll_speed = float(p.get("scroll_speed", 1.0))
        self.conf_min = float(p.get("conf_min", 0.35))
        self.lost_release_ms = float(p.get("lost_release_ms", 220))

    def reset(self) -> None:
        if self._down:
            winput.button_up(self._down)
        self._down = None
        self._anchor = None
        self._frozen = False
        self._scrolling = False
        self._scroll_accum = 0.0
        self._pinch = self._pinch_middle = None
        self.left.reset()
        self.right.reset()
        self.filter.reset()
        self.rope.reset()
        self._trail.clear()
        self.debug.update(down=None, frozen=False)

    # ---------------------------------------------------------------- mapping

    def screen_rect(self) -> tuple[int, int, int, int]:
        return winput.primary_screen() if self.screen == "primary" else winput.virtual_screen()

    def _region_uv(self, nx: float, ny: float) -> tuple[float, float]:
        """Image coordinates to 0..1 across the calibrated region, flipped."""
        x0, y0, x1, y1 = self.region
        u = (nx - x0) / max(x1 - x0, 1e-3)
        v = (ny - y0) / max(y1 - y0, 1e-3)
        if self.flip_x:
            u = 1.0 - u
        if self.flip_y:
            v = 1.0 - v
        return min(1.0, max(0.0, u)), min(1.0, max(0.0, v))

    def _to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        u, v = self._region_uv(nx, ny)
        left, top, width, height = self.screen_rect()
        return left + u * (width - 1), top + v * (height - 1)

    def _anchor_from_trail(self, t: float, current: tuple[float, float]
                           ) -> tuple[float, float]:
        """Where the pointer was just before the fingers started closing.

        Bounded, though. The lookback exists to undo the small drift a pinch
        causes, and that drift is a handful of pixels. If the trail says the
        pointer was somewhere far away, the hand was simply travelling -- and
        honouring that would yank the cursor backwards to a place the user had
        already left, which is much worse than the drift it was correcting.
        """
        target = t - self.lookback_ms / 1000.0
        best = None
        for stamp, sx, sy in self._trail:
            if stamp <= target:
                best = (sx, sy)
            else:
                break
        if best is None:
            return current

        limit = max(self.freeze_px * 2.5, 12.0)
        dx = best[0] - current[0]
        dy = best[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance <= limit:
            return best
        scale = limit / distance
        return current[0] + dx * scale, current[1] + dy * scale

    def stillness(self) -> tuple[float, float]:
        """RMS wobble over the last half second, in screen pixels, before and
        after filtering. Meaningful while the hand is held still; it is the
        number to watch when comparing settings."""
        try:
            samples = list(self._stillness)
        except RuntimeError:
            return 0.0, 0.0
        if len(samples) < 10:
            return 0.0, 0.0

        def rms(ix: int, iy: int) -> float:
            mx = sum(s[ix] for s in samples) / len(samples)
            my = sum(s[iy] for s in samples) / len(samples)
            return math.sqrt(sum((s[ix] - mx) ** 2 + (s[iy] - my) ** 2
                                 for s in samples) / len(samples))
        return rms(0, 1), rms(2, 3)

    # ----------------------------------------------------------------- update

    def update(self, hand, now: float, t_sample: float | None = None) -> None:
        """``now`` is the PC clock, for timeouts; ``t_sample`` is the phone's
        capture timestamp, for the filter. Receive times arrive in bursts -- two
        frames from one USB read share a timestamp -- while capture times are
        exactly one frame apart, and the filter's speed estimate is only as good
        as its time base."""
        if not hand.ok or hand.quality < self.conf_min:
            # Releasing on the very first missing frame would turn a momentary
            # tracking gap in the middle of a drag into a dropped file.
            lost_ms = (now - self._last_good) * 1000.0
            if self._down and lost_ms > self.lost_release_ms:
                winput.button_up(self._down)
                self._down = None
                self._frozen = False
                self.debug.update(down=None, frozen=False)
            if lost_ms > self.lost_release_ms:
                self.filter.reset()
                self.rope.reset()
                self._pinch = self._pinch_middle = None
                self._stillness.clear()
            return
        self._last_good = now

        t = t_sample if t_sample and t_sample > 0 else now
        if self._last_t is not None and t < self._last_t:
            # The phone's clock restarted (a reconnect): start the filter afresh
            # rather than feed it a negative interval.
            self.filter.reset()
            self.rope.reset()
        self._last_t = t

        nx, ny = hand.anchor(self.anchor)
        self.debug["anchor"] = (nx, ny)
        u, v = self._region_uv(nx, ny)
        left, top, width, height = self.screen_rect()
        ref_w = REF_HEIGHT * width / max(height, 1)

        fx, fy = self.filter(u * ref_w, v * REF_HEIGHT, t)
        if self.lead_ms > 0:
            vx, vy = self.filter.velocity
            fx += vx * self.lead_ms / 1000.0
            fy += vy * self.lead_ms / 1000.0
        rx, ry = self.rope(fx, fy)
        sx = left + rx / ref_w * (width - 1)
        sy = top + ry / REF_HEIGHT * (height - 1)
        self._trail.append((t, sx, sy))
        self._stillness.append((left + u * (width - 1), top + v * (height - 1), sx, sy))

        # Lightly smoothed: one noisy frame at the threshold should not be a
        # click, and half-weighting costs about one frame of response.
        p = hand.pinch_index
        self._pinch = p if self._pinch is None else self._pinch + 0.5 * (p - self._pinch)
        pm = hand.pinch_middle
        self._pinch_middle = pm if self._pinch_middle is None else (
            self._pinch_middle + 0.5 * (pm - self._pinch_middle))
        self.debug["pinch"] = self._pinch

        if self._handle_scroll(hand, sy):
            return

        left_pinched = self.left.update(self._pinch) if self.click else False
        # Only let thumb-to-middle count when the index is clearly *not* the
        # finger being pinched. Otherwise a normal left pinch, where the middle
        # finger often trails along behind the index, fires a right click too.
        use_right = self.click and self.right_click
        right_candidate = (self._pinch_middle if use_right and
                           self._pinch_middle < self._pinch else 1.0)
        right_pinched = self.right.update(right_candidate) if use_right else False

        wanted = "left" if left_pinched else ("right" if right_pinched else None)

        if wanted != self._down:
            if self._down:
                winput.button_up(self._down)
                self._down = None
                self._frozen = False
                self._anchor = None
                self.debug.update(down=None, frozen=False)
            if wanted:
                ax, ay = self._anchor_from_trail(t, (sx, sy))
                self._anchor = (ax, ay)
                self._frozen = True
                # Move to the anchor before pressing, so the click lands where
                # the pointer was aimed rather than where the pinch dragged it.
                winput.move_to(int(round(ax)), int(round(ay)))
                winput.button_down(wanted)
                self._down = wanted
                self.debug.update(down=wanted, frozen=True)
                return

        if self._down and self._frozen and self._anchor:
            ax, ay = self._anchor
            if abs(sx - ax) > self.freeze_px or abs(sy - ay) > self.freeze_px:
                # Deliberate movement: stop holding the pointer still and let
                # this become a drag.
                self._frozen = False
                self.debug["frozen"] = False
            else:
                return

        winput.move_to(int(round(sx)), int(round(sy)))

    def _handle_scroll(self, hand, sy: float) -> bool:
        if not self.scroll_enabled:
            return False
        two_fingers = (hand.ext[1] > 0.62 and hand.ext[2] > 0.62
                       and hand.ext[3] < 0.30 and hand.ext[4] < 0.30)
        if not two_fingers:
            self._scrolling = False
            return False
        if not self._scrolling:
            self._scrolling = True
            self._scroll_ref = sy
            self._scroll_accum = 0.0
            return True
        delta = (self._scroll_ref - sy) * self.scroll_speed
        self._scroll_ref = sy
        self._scroll_accum += delta
        # One wheel notch per 40 px of hand travel, which is roughly the same
        # ratio a real wheel gives per detent.
        while abs(self._scroll_accum) >= 40.0:
            step = 1 if self._scroll_accum > 0 else -1
            winput.wheel(step)
            self._scroll_accum -= step * 40.0
        return True


class Calibrator:
    """Watches where the hand goes for a few seconds and turns that into the
    region of the camera's view that maps onto the screen."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self.duration = 6.0
        self._until = 0.0
        self._bounds: list[float] | None = None

    def start(self, duration: float = 6.0) -> None:
        with self._lock:
            self.active = True
            self.duration = duration
            self._until = time.monotonic() + duration
            self._bounds = None

    def cancel(self) -> None:
        with self._lock:
            self.active = False

    def add(self, x: float, y: float) -> None:
        with self._lock:
            if not self.active:
                return
            b = self._bounds
            if b is None:
                self._bounds = [x, y, x, y]
            else:
                b[0], b[1] = min(b[0], x), min(b[1], y)
                b[2], b[3] = max(b[2], x), max(b[3], y)

    def poll(self) -> tuple[bool, list[float] | None]:
        """(finished just now, region or None if there was too little movement)."""
        with self._lock:
            if not self.active or time.monotonic() < self._until:
                return False, None
            self.active = False
            b = self._bounds
        if b is None or (b[2] - b[0]) < 0.05 or (b[3] - b[1]) < 0.05:
            return True, None
        # A small inset rather than the raw extremes: the very edge of a sweep is
        # where you were already turning around, so it is the least repeatable
        # part of the range and would make the screen corners hard to reach.
        mx = (b[2] - b[0]) * 0.06
        my = (b[3] - b[1]) * 0.06
        return True, [round(max(0.0, b[0] + mx), 4), round(max(0.0, b[1] + my), 4),
                      round(min(1.0, b[2] - mx), 4), round(min(1.0, b[3] - my), 4)]

    def state(self) -> dict | None:
        with self._lock:
            if not self.active:
                return None
            remaining = max(0.0, self._until - time.monotonic())
            return {"remaining": round(remaining, 2),
                    "progress": round(1.0 - remaining / self.duration, 3),
                    "bounds": list(self._bounds) if self._bounds else None}


class HotkeyListener(threading.Thread):
    """Global hotkeys, in their own thread with its own message loop.

    Worth the ctypes: pointer mode can put the machine in a state where clicking
    a Stop button is exactly the thing you cannot reliably do.
    """

    WM_HOTKEY = 0x0312
    WM_QUIT = 0x0012
    MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 1, 2, 4, 8, 0x4000

    def __init__(self, specs: dict, callback) -> None:
        super().__init__(name="handpilot-hotkeys", daemon=True)
        self._specs = specs
        self._callback = callback
        self._thread_id = 0
        self.error = ""

    @classmethod
    def _parse(cls, spec: str):
        parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
        mods = 0
        key = None
        for part in parts:
            if part in ("ctrl", "control"):
                mods |= cls.MOD_CONTROL
            elif part == "alt":
                mods |= cls.MOD_ALT
            elif part == "shift":
                mods |= cls.MOD_SHIFT
            elif part in ("win", "cmd", "meta"):
                mods |= cls.MOD_WIN
            else:
                key = part
        vk = winput.VK.get(key) if key else None
        return (mods | cls.MOD_NOREPEAT, vk) if vk else None

    def run(self) -> None:
        user32 = winput.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()

        user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND,
                                       wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int

        registered = {}
        for index, (name, spec) in enumerate(self._specs.items(), start=1):
            if not spec:
                continue
            parsed = self._parse(spec)
            if not parsed:
                self.error = f"unknown hotkey '{spec}'"
                continue
            mods, vk = parsed
            if user32.RegisterHotKey(None, index, mods, vk):
                registered[index] = name
            else:
                # Almost always another application already owns the key.
                self.error = f"{spec} is already taken by another app"

        message = MSG()
        try:
            while True:
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result in (0, -1):
                    break
                if message.message == self.WM_HOTKEY:
                    name = registered.get(int(message.wParam))
                    if name:
                        try:
                            self._callback(name)
                        except Exception:                 # noqa: BLE001
                            pass
        finally:
            for index in registered:
                user32.UnregisterHotKey(None, index)

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, self.WM_QUIT, 0, 0)


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                ("wParam", ctypes.c_size_t), ("lParam", ctypes.c_ssize_t),
                ("time", wintypes.DWORD), ("pt_x", wintypes.LONG), ("pt_y", wintypes.LONG)]


class Controller:
    """Owns the link, the modes, calibration and the raw-data outputs."""

    MODES = ("off", "pointer", "gestures")

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.mode = "off"
        self.pointer = PointerMode()
        self.engine = GestureEngine()
        self.calibrator = Calibrator()
        self.link = Link(self._on_frame)
        self.hotkeys: HotkeyListener | None = None
        self.forwarder: UdpForwarder | None = None
        #: Called (from whichever thread finished it) after calibration changes
        #: the config, so the owner can persist it.
        self.on_config_changed = None

        self.latest_frame = None
        self.latest_hand = None
        self.log_lines: deque = deque(maxlen=300)
        self.log_rev = 0
        self._frame_lock = threading.Lock()
        self._process_ms = 0.0

        self.apply_config()

    # ------------------------------------------------------------------ config

    def apply_config(self) -> None:
        self.pointer.configure(self.cfg["pointer"])
        g = self.cfg["gestures"]
        self.engine.cooldown_ms = float(g.get("cooldown_ms", 450))
        self.engine.swipes.config.window_ms = float(g.get("swipe_window_ms", 260))
        self.engine.swipes.config.min_travel = float(g.get("swipe_travel", 1.7))
        self.engine.swipes.config.require_open = bool(g.get("swipe_require_open", True))
        for gid, raw in g.get("bindings", {}).items():
            if gid in self.engine.bindings:
                self.engine.bindings[gid] = Binding(
                    enabled=bool(raw.get("enabled", False)),
                    action=str(raw.get("action", "none")),
                    mode=str(raw.get("mode", "once")),
                    hold_ms=float(raw.get("hold_ms", 180)),
                    repeat_ms=float(raw.get("repeat_ms", 200)),
                    param=str(raw.get("param", "")),
                )
        self._configure_forwarder()
        if self.mode == "gestures" and not self.gestures_enabled:
            self.set_mode("off")

    def _configure_forwarder(self) -> None:
        stream = self.cfg.get("stream", {})
        target = stream.get("udp_target", "127.0.0.1:9872")
        if not stream.get("udp"):
            if self.forwarder:
                self.forwarder.close()
                self.forwarder = None
            return
        try:
            from .stream import parse_target
            wanted = parse_target(target)
        except ValueError as exc:
            self.log(f"UDP output: {exc}")
            return
        if self.forwarder is None or self.forwarder.target != wanted:
            if self.forwarder:
                self.forwarder.close()
            self.forwarder = UdpForwarder(target)
            self.log(f"UDP output: sending frames to {target}")

    @property
    def gestures_enabled(self) -> bool:
        return bool(self.cfg.get("ui", {}).get("experimental_gestures", False))

    def send_phone(self, settings: dict) -> None:
        """Change camera settings on the phone. It applies them, remembers them,
        and reports them back in its next status message."""
        self.link.send_command(settings)

    # ------------------------------------------------------------------- link

    def connect(self) -> None:
        self.link.start(self.cfg["transport"], self.cfg["wifi_host"], int(self.cfg["port"]))
        how = "USB" if self.cfg["transport"] == "usb" else f"Wi-Fi ({self.cfg['wifi_host']})"
        self.log(f"connecting over {how}, port {self.cfg['port']}")

    def disconnect(self) -> None:
        self.set_mode("off")
        self.link.stop()
        self.log("disconnected")

    def start_hotkeys(self) -> None:
        specs = {k: v for k, v in self.cfg.get("hotkeys", {}).items() if v}
        if not specs:
            return
        self.hotkeys = HotkeyListener(specs, self._on_hotkey)
        self.hotkeys.start()

    def shutdown(self) -> None:
        self.set_mode("off")
        self.link.stop()
        if self.hotkeys:
            self.hotkeys.stop()
        if self.forwarder:
            self.forwarder.close()
        winput.release_all_buttons()

    # ------------------------------------------------------------------ modes

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES or mode == self.mode:
            return
        if mode == "gestures" and not self.gestures_enabled:
            return
        # Leaving pointer mode with a button held would hand the desktop back in
        # a state where every move is a drag.
        self.pointer.reset()
        self.engine.reset()
        self.mode = mode
        self.cfg["mode"] = mode
        self.log(f"mode: {mode}")

    def cycle_mode(self) -> None:
        order = ("off", "pointer", "gestures") if self.gestures_enabled else ("off", "pointer")
        current = order.index(self.mode) if self.mode in order else 0
        self.set_mode(order[(current + 1) % len(order)])

    def _on_hotkey(self, name: str) -> None:
        if name == "panic":
            self.set_mode("off")
            winput.release_all_buttons()
            self.log("panic key: control released")
        elif name == "cycle_mode":
            self.cycle_mode()

    # ------------------------------------------------------------ calibration

    def start_calibration(self) -> None:
        self.calibrator.start()
        self.log("calibrating: move your hand around the area you want to use")

    def tick_calibration(self) -> None:
        finished, region = self.calibrator.poll()
        if not finished:
            return
        if region is None:
            self.log("calibration: not enough movement, area unchanged")
            return
        self.cfg["pointer"]["region"] = region
        self.cfg["pointer"]["needs_calibration"] = False
        self.pointer.configure(self.cfg["pointer"])
        self.log(f"calibration: area set to {region}")
        if self.on_config_changed:
            self.on_config_changed()

    # ----------------------------------------------------------------- frames

    def _on_frame(self, frame) -> None:
        started = time.perf_counter()
        hand = handmodel.build(frame, conf_min=self.cfg["pointer"]["conf_min"])
        with self._frame_lock:
            self.latest_frame = frame
            self.latest_hand = hand

        if self.forwarder:
            self.forwarder.send(frame)
        if self.calibrator.active:
            if hand.ok:
                self.calibrator.add(*hand.anchor(self.pointer.anchor))
            self.tick_calibration()

        if self.mode == "pointer":
            self.pointer.update(hand, frame.t_recv, frame.t_capture)
        elif self.mode == "gestures":
            fired = self.engine.update(hand, frame.t_recv,
                                       flip_x=self.pointer.flip_x,
                                       flip_y=self.pointer.flip_y)
            for gid, binding in fired:
                actions.run(binding.action, self, binding.param)
                spec = gestures.GESTURE_BY_ID.get(gid)
                self.log(f"{spec.label if spec else gid} -> {actions.label_for(binding.action)}")

        elapsed = (time.perf_counter() - started) * 1000.0
        self._process_ms = self._process_ms * 0.95 + elapsed * 0.05

    # ------------------------------------------------------------------ readouts

    def snapshot(self):
        with self._frame_lock:
            return self.latest_frame, self.latest_hand

    @property
    def latency_estimate_ms(self) -> float:
        """Sensor to synthetic input, in milliseconds.

        The phone measures its own capture-to-send on one clock, the echo
        measures the cable on ours, and half a round trip is the standard
        one-way estimate; add what this side spends per frame.
        """
        phone = float(self.link.stats.phone.get("inference_ms", 0.0))
        frame = self.latest_frame
        pipeline = frame.phone_ms if frame is not None else phone
        return pipeline + self.link.stats.rtt_ms / 2.0 + self._process_ms

    @property
    def process_ms(self) -> float:
        return self._process_ms

    def log(self, text: str) -> None:
        self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {text}")
        self.log_rev += 1

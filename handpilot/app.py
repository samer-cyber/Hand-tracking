"""The app behind the window: settings, commands, and the live view's data.

The window (server.py and web/) is a thin client of this class. Every setting
change arrives here as a dotted path and a value, is checked against SCHEMA,
applied, and saved -- so a bad request can never put a string where the
filter expects a number. Camera settings ("phone.*") are not stored here at
all; they are sent to the phone, which applies and remembers them.
"""

from __future__ import annotations

import threading
import time

from . import __version__, actions, config, gestures, usbmux, winput
from .controller import Controller
from .stream import parse_target

_BOOL = ("bool",)

SCHEMA: dict[str, tuple] = {
    "transport": ("choice", ("usb", "wifi")),
    "wifi_host": ("str", 100),
    "port": ("int", 1, 65535),
    "autoconnect": _BOOL,
    "pointer.region": ("region",),
    "pointer.anchor": ("choice", ("knuckles", "pinch")),
    "pointer.steadiness": ("float", 0.0, 1.0),
    "pointer.custom_filter": _BOOL,
    "pointer.min_cutoff": ("float", 0.05, 10.0),
    "pointer.beta": ("float", 0.0, 0.5),
    "pointer.deadzone_px": ("float", 0.0, 10.0),
    "pointer.lead_ms": ("float", 0.0, 60.0),
    "pointer.flip_x": _BOOL,
    "pointer.flip_y": _BOOL,
    "pointer.screen": ("choice", ("primary", "all")),
    "pointer.click": _BOOL,
    "pointer.pinch_on": ("float", 0.05, 0.8),
    "pointer.pinch_off": ("float", 0.06, 1.2),
    "pointer.click_freeze_px": ("float", 0.0, 120.0),
    "pointer.anchor_lookback_ms": ("float", 0.0, 250.0),
    "pointer.right_click": _BOOL,
    "pointer.right_pinch_on": ("float", 0.05, 0.8),
    "pointer.right_pinch_off": ("float", 0.06, 1.2),
    "pointer.scroll_enabled": _BOOL,
    "pointer.scroll_speed": ("float", 0.2, 5.0),
    "pointer.conf_min": ("float", 0.05, 0.9),
    "pointer.lost_release_ms": ("float", 50.0, 1000.0),
    "stream.udp": _BOOL,
    "stream.udp_target": ("target",),
    "stream.browser": _BOOL,
    "ui.experimental_gestures": _BOOL,
    "gestures.cooldown_ms": ("float", 0.0, 3000.0),
}

BINDING_FIELDS: dict[str, tuple] = {
    "enabled": _BOOL,
    "action": ("choice", tuple(actions.ACTIONS)),
    "mode": ("choice", ("once", "repeat")),
    "hold_ms": ("float", 0.0, 3000.0),
    "repeat_ms": ("float", 50.0, 3000.0),
    "param": ("str", 300),
}

#: What the phone accepts, mirroring SettingsStore.applyCommand in the iOS app.
PHONE: dict[str, tuple] = {
    "fps": ("choice", (60, 120, 240)),
    "shutter": ("choice", (120, 250, 300, 360, 500, 750, 1000)),
    "camera": ("choice", ("back", "front")),
    "crop": _BOOL,
    "iso_auto": _BOOL,
    "iso_ceiling": ("float", 200.0, 12800.0),
    "zoom": ("float", 1.0, 3.0),
    "focus_auto": _BOOL,
    "lens": ("float", 0.0, 1.0),
    "orientation": ("choice", (-1, 0, 1, 2, 3)),
    "preview": _BOOL,
}

_RECONNECT_ON = {"transport", "wifi_host", "port"}


def friendly_error(text: str) -> str:
    """Socket errors as instructions. The USB path already speaks plainly; these
    are the raw OS messages a Wi-Fi connection produces."""
    lowered = (text or "").lower()
    if not lowered:
        return ""
    if "10061" in lowered or "refused" in lowered:
        return "The iPhone didn't answer. Open HandPilot on it and tap Start."
    if "10060" in lowered or "timed out" in lowered or "10065" in lowered:
        return ("No answer at that address. Check the iPhone's IP, and that both "
                "devices are on the same Wi-Fi.")
    if "closed the connection" in lowered or "10054" in lowered or "reset" in lowered:
        return "The iPhone app stopped streaming. Tap Start on the phone again."
    return text


def _coerce(spec: tuple, value):
    kind = spec[0]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError("expected true or false")
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a number")
        return float(min(spec[2], max(spec[1], value)))
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("expected a whole number")
        return int(min(spec[2], max(spec[1], round(value))))
    if kind == "str":
        if not isinstance(value, str) or len(value) > spec[1]:
            raise ValueError(f"expected text up to {spec[1]} characters")
        return value.strip()
    if kind == "choice":
        # bool is an int in Python, so True would otherwise match the choice 1.
        if not isinstance(value, bool):
            for option in spec[1]:
                if value == option:
                    return option
        raise ValueError(f"expected one of {', '.join(map(str, spec[1]))}")
    if kind == "region":
        if (not isinstance(value, list) or len(value) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
            raise ValueError("expected [left, top, right, bottom]")
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in value)
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            raise ValueError("region is too small")
        return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]
    if kind == "target":
        parse_target(str(value))
        return str(value).strip()
    raise ValueError("unknown setting type")


class App:
    def __init__(self, cfg: dict, config_path: str | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.controller = Controller(cfg)
        self.controller.on_config_changed = self.schedule_save
        self.quit_event = threading.Event()
        self._save_timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------------------- lifecycle

    def start(self, hotkeys: bool = True) -> None:
        if hotkeys:
            self.controller.start_hotkeys()
        if self.cfg.get("autoconnect", True):
            self.controller.connect()
        if self.cfg["pointer"].get("needs_calibration"):
            self.controller.log("press Calibrate once so the screen area matches your reach")

    def shutdown(self) -> None:
        with self._lock:
            if self._save_timer:
                self._save_timer.cancel()
        self.save_now()
        self.controller.shutdown()

    def schedule_save(self) -> None:
        # Debounced: dragging a slider would otherwise write the file on every
        # step of travel.
        with self._lock:
            if self._save_timer:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(0.7, self.save_now)
            self._save_timer.daemon = True
            self._save_timer.start()

    def save_now(self) -> None:
        try:
            config.save(self.cfg, self.config_path)
        except OSError as exc:
            self.controller.log(f"could not save settings: {exc}")

    # --------------------------------------------------------------- settings

    def _spec(self, path: str) -> tuple:
        if path in SCHEMA:
            return SCHEMA[path]
        parts = path.split(".")
        if (len(parts) == 4 and parts[:2] == ["gestures", "bindings"]
                and parts[2] in gestures.GESTURE_BY_ID and parts[3] in BINDING_FIELDS):
            return BINDING_FIELDS[parts[3]]
        raise KeyError(path)

    def set_setting(self, path: str, value) -> tuple[bool, str]:
        """Apply one change. Returns (ok, error message)."""
        try:
            if path.startswith("phone."):
                key = path[6:]
                if key not in PHONE:
                    return False, f"unknown phone setting {key!r}"
                coerced = _coerce(PHONE[key], value)
                if not self.controller.link.stats.connected:
                    return False, "connect the iPhone to change camera settings"
                if isinstance(coerced, (int, float)) and not isinstance(coerced, bool) \
                        and key != "orientation":
                    coerced = float(coerced)
                self.controller.send_phone({key: coerced})
                return True, ""
            coerced = _coerce(self._spec(path), value)
        except KeyError:
            return False, f"unknown setting {path!r}"
        except ValueError as exc:
            return False, f"{path}: {exc}"

        node = self.cfg
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        previous = node.get(parts[-1])
        node[parts[-1]] = coerced

        pointer = self.cfg["pointer"]
        # Keep each hysteresis band open: a release threshold at or below the
        # press threshold would turn a held pinch into a stream of clicks.
        if path in ("pointer.pinch_on", "pointer.pinch_off"):
            pointer["pinch_off"] = max(pointer["pinch_off"], round(pointer["pinch_on"] + 0.04, 3))
        if path in ("pointer.right_pinch_on", "pointer.right_pinch_off"):
            pointer["right_pinch_off"] = max(pointer["right_pinch_off"],
                                             round(pointer["right_pinch_on"] + 0.04, 3))
        if path == "pointer.anchor" and previous != coerced:
            pointer["needs_calibration"] = True
            self.controller.log("tracking point changed: press Calibrate so the screen area fits")

        self.controller.apply_config()
        if path in _RECONNECT_ON and previous != coerced:
            self.controller.link.stop()
            self.controller.connect()
        self.schedule_save()
        return True, ""

    # --------------------------------------------------------------- commands

    def command(self, name: str, value=None) -> dict:
        c = self.controller
        if name == "connect":
            c.link.stop()
            c.connect()
        elif name == "disconnect":
            c.disconnect()
        elif name == "mode":
            if value not in Controller.MODES:
                return {"ok": False, "error": "unknown mode"}
            if value == "gestures" and not c.gestures_enabled:
                return {"ok": False, "error": "turn on experimental gestures first"}
            c.set_mode(value)
        elif name == "calibrate":
            if not c.link.stats.connected:
                return {"ok": False, "error": "connect the iPhone first"}
            c.start_calibration()
        elif name == "cancel_calibration":
            c.calibrator.cancel()
            c.log("calibration cancelled")
        elif name == "panic":
            c.set_mode("off")
            winput.release_all_buttons()
        elif name == "reset_pointer":
            keep = {k: self.cfg["pointer"][k] for k in ("region", "needs_calibration")}
            self.cfg["pointer"] = {**config.DEFAULTS["pointer"], **keep}
            c.apply_config()
            self.schedule_save()
            c.log("pointer settings reset to defaults (calibrated area kept)")
        elif name == "diagnose":
            return {"ok": True, "checks": self.diagnose()}
        elif name == "quit":
            self.quit_event.set()
        else:
            return {"ok": False, "error": f"unknown command {name!r}"}
        return {"ok": True}

    def diagnose(self) -> list[dict]:
        """The connection path, one link at a time, each with the specific fix.

        "No Apple service", "no cable", "phone locked" and "app not started" all
        look identical from the outside -- it just does not connect."""
        checks = []

        def add(ok: bool, text: str, fix: str = "") -> None:
            checks.append({"ok": ok, "text": text, "fix": fix})

        stats = self.controller.link.stats
        if self.cfg["transport"] == "usb":
            try:
                devices = usbmux.list_devices(timeout=1.5)
            except usbmux.UsbmuxError as exc:
                add(False, "Apple's USB service is not reachable", str(exc))
                return checks
            add(True, "Apple's USB service is running")
            usb = [d for d in devices if d["usb"]]
            if not devices:
                add(False, "No iPhone found on USB",
                    "Plug it in with a data cable, unlock it, and tap Trust if it asks.")
                return checks
            if not usb:
                add(False, "The iPhone is only visible over Wi-Fi sync",
                    "Use a data-capable cable; some cables only charge.")
                return checks
            add(True, "iPhone attached over USB")
        elif not self.cfg.get("wifi_host"):
            add(False, "No phone IP address set",
                "Enter the iPhone's Wi-Fi IP in Advanced > Connection.")
            return checks

        if stats.connected:
            add(True, "Connected to HandPilot on the iPhone")
            if stats.fps < 1:
                add(False, "Connected, but no frames are arriving",
                    "Tap Start in the iPhone app.")
            else:
                add(True, f"Receiving {stats.fps:.0f} frames per second")
        else:
            add(False, "HandPilot on the iPhone is not answering",
                stats.error or "Open HandPilot on the iPhone and tap Start. "
                               "The port there must match the one here.")
        return checks

    # -------------------------------------------------------------- read side

    def state(self) -> dict:
        return {
            "version": __version__,
            "config": self.cfg,
            "config_path": self.config_path or config.CONFIG_PATH,
            "actions": [{"id": a.id, "label": actions.label_for(a.id), "param": a.takes_param}
                        for a in (actions.ACTIONS[i] for i in actions.ordered_ids())],
            "gestures": [{"id": g.id, "label": g.label, "hint": g.hint}
                         for g in gestures.GESTURES],
        }

    def live(self) -> dict:
        """One snapshot for the live view. Built per call (30 times a second per
        open window), from values the link thread is updating concurrently --
        each is read once, so a snapshot can be a frame stale but never torn."""
        c = self.controller
        c.tick_calibration()
        frame, hand = c.snapshot()
        stats = c.link.stats
        connected = stats.connected
        raw, out = c.pointer.stillness() if c.mode == "pointer" else (0.0, 0.0)

        view = None
        if connected and frame is not None and frame.hand and len(frame.x) == 21:
            view = {
                "w": frame.width, "h": frame.height,
                "x": [round(v, 4) for v in frame.x],
                "y": [round(v, 4) for v in frame.y],
                "c": [round(v, 2) for v in frame.conf],
                "usable": bool(hand is not None and hand.ok),
                "age": round((time.perf_counter() - frame.t_recv) * 1000.0),
            }

        anchor = None
        pinch = None
        if hand is not None and hand.ok and view is not None:
            anchor = [round(v, 4) for v in hand.anchor(c.pointer.anchor)]
            pinch = round(c.pointer.debug["pinch"] if c.mode == "pointer" else hand.pinch_index, 3)

        return {
            "conn": {
                "connected": connected,
                "transport": stats.transport,
                "device": stats.device,
                "error": friendly_error(stats.error),
                "fps": round(stats.fps, 1),
                "rtt": round(stats.rtt_ms, 2),
            },
            "phone": stats.phone if connected else {},
            "mode": c.mode,
            "latency": round(c.latency_estimate_ms, 1) if connected and stats.fps > 0 else None,
            "process_ms": round(c.process_ms, 3),
            "frame": view,
            "pointer": {
                "down": c.pointer.debug.get("down"),
                "anchor": anchor,
                "pinch": pinch,
                "stillness_raw": round(raw, 2),
                "stillness": round(out, 2),
            },
            "calib": c.calibrator.state(),
            "pose": c.engine.active_pose if c.mode == "gestures" else None,
            "needs_calibration": bool(self.cfg["pointer"].get("needs_calibration")),
            "log_rev": c.log_rev,
        }

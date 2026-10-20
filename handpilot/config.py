"""Settings, with defaults that are meant to be usable before you touch anything.

Camera settings are not here. They live on the phone, which remembers them, and
the desktop only ever asks it to change them -- two copies of the same setting
is how a phone ends up silently reverting whatever you just picked on it.

Stored as ``config.json``: next to ``run.py`` when running from a checkout, in
the user's app-data folder when installed, or wherever ``HANDPILOT_CONFIG``
points. Unknown keys in the file are kept and missing ones are filled from the
defaults, so an old config keeps working after new settings are added.
"""

from __future__ import annotations

import copy
import json
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT_DIR = os.path.dirname(_PACKAGE_DIR)


def _default_path() -> str:
    override = os.environ.get("HANDPILOT_CONFIG")
    if override:
        return override
    if os.path.exists(os.path.join(_CHECKOUT_DIR, "run.py")):
        return os.path.join(_CHECKOUT_DIR, "config.json")
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "HandPilot", "config.json")


CONFIG_PATH = _default_path()
VERSION = 2

DEFAULTS: dict = {
    "version": VERSION,
    "transport": "usb",
    "wifi_host": "",
    "port": 9871,
    "autoconnect": True,
    "mode": "off",

    "pointer": {
        # Which part of the camera's view maps onto the screen, as
        # (left, top, right, bottom) in normalised image coordinates. The default
        # is a middle box rather than the whole frame: reaching the physical
        # corners of the camera's view means fully extending your arm, and the
        # corners are also where the lens distorts most.
        "region": [0.18, 0.15, 0.82, 0.85],
        # What the cursor follows. "knuckles" is the average of the four
        # knuckles: the steadiest points the model reports, and they barely move
        # when you pinch. "pinch" is the midpoint of thumb and index tips, which
        # feels more direct but inherits the thumb tip's noise.
        "anchor": "knuckles",
        # The one smoothing control: 0 is most responsive, 1 is steadiest. It
        # drives the One Euro filter and the still-hand dead zone together.
        "steadiness": 0.5,
        # Expert override of the three values `steadiness` normally sets.
        "custom_filter": False,
        "min_cutoff": 0.7,
        "beta": 0.0095,
        "deadzone_px": 1.8,
        # Extrapolation along the current velocity. Off by default: it buys a
        # few milliseconds and pays for them with an overshoot every time the
        # hand stops.
        "lead_ms": 0.0,
        # The camera faces you, so it sees you the way another person would:
        # your right hand appears on its left. Without this the cursor goes the
        # wrong way and nothing else can be judged.
        "flip_x": True,
        "flip_y": False,
        # "primary" maps the region onto the main monitor; "all" onto the whole
        # virtual desktop, which doubles the gain on a two-monitor setup.
        "screen": "primary",
        "click": True,
        "pinch_on": 0.30,
        "pinch_off": 0.44,
        # How far the pointer may wander during a pinch and still count as a
        # click rather than a drag. In screen pixels.
        "click_freeze_px": 16,
        # The pointer is sampled from slightly before the pinch was detected,
        # because the fingers are already moving together by the time it is.
        "anchor_lookback_ms": 90,
        "right_click": True,
        "right_pinch_on": 0.30,
        "right_pinch_off": 0.44,
        "scroll_enabled": False,
        "scroll_speed": 1.0,
        "conf_min": 0.35,
        # How long the hand may be missing before any held button is released.
        "lost_release_ms": 220,
        # Until the first calibration (or after the tracking point changes) the
        # window asks for one: the default region is only a guess at your reach.
        "needs_calibration": True,
    },

    # Raw landmarks for other programs. Both are off by default: anything that
    # can reach the port can watch your hand.
    "stream": {
        "udp": False,
        "udp_target": "127.0.0.1:9872",
        "browser": False,
    },

    "ui": {
        "port": 9870,
        "experimental_gestures": False,
    },

    "gestures": {
        "cooldown_ms": 450,
        "swipe_window_ms": 260,
        "swipe_travel": 1.7,
        "swipe_require_open": True,
        "bindings": {
            "thumbs_up": {"enabled": True, "action": "volume_up", "mode": "repeat",
                          "hold_ms": 200, "repeat_ms": 180, "param": ""},
            "thumbs_down": {"enabled": True, "action": "volume_down", "mode": "repeat",
                            "hold_ms": 200, "repeat_ms": 180, "param": ""},
            "fist": {"enabled": True, "action": "media_play_pause", "mode": "once",
                     "hold_ms": 220, "repeat_ms": 200, "param": ""},
            "swipe_right": {"enabled": True, "action": "media_next", "mode": "once",
                            "hold_ms": 0, "repeat_ms": 200, "param": ""},
            "swipe_left": {"enabled": True, "action": "media_prev", "mode": "once",
                           "hold_ms": 0, "repeat_ms": 200, "param": ""},
            "peace": {"enabled": False, "action": "screenshot_region", "mode": "once",
                      "hold_ms": 260, "repeat_ms": 200, "param": ""},
            "ok_sign": {"enabled": False, "action": "click_left", "mode": "once",
                        "hold_ms": 160, "repeat_ms": 200, "param": ""},
            "open_palm": {"enabled": False, "action": "mode_cycle", "mode": "once",
                          "hold_ms": 400, "repeat_ms": 200, "param": ""},
            "point": {"enabled": False, "action": "none", "mode": "once",
                      "hold_ms": 200, "repeat_ms": 200, "param": ""},
            "pinch": {"enabled": False, "action": "click_left", "mode": "once",
                      "hold_ms": 120, "repeat_ms": 200, "param": ""},
            "rock": {"enabled": False, "action": "none", "mode": "once",
                     "hold_ms": 220, "repeat_ms": 200, "param": ""},
            "three": {"enabled": False, "action": "none", "mode": "once",
                      "hold_ms": 220, "repeat_ms": 200, "param": ""},
            "shaka": {"enabled": False, "action": "none", "mode": "once",
                      "hold_ms": 220, "repeat_ms": 200, "param": ""},
            "swipe_up": {"enabled": False, "action": "scroll_up", "mode": "once",
                         "hold_ms": 0, "repeat_ms": 200, "param": ""},
            "swipe_down": {"enabled": False, "action": "scroll_down", "mode": "once",
                           "hold_ms": 0, "repeat_ms": 200, "param": ""},
        },
    },

    # Global, so the pointer can always be stopped without needing to click a
    # button with the pointer you are trying to stop.
    "hotkeys": {
        "panic": "f9",
        "cycle_mode": "f10",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _migrate(raw: dict) -> dict:
    """Bring a version-1 file (the Tkinter app's) up to date.

    Some values from that era are reset rather than carried over: its
    "Smoothing" slider was inverted, so saved filter values are usually the
    opposite of what was intended; its lead-ahead overshot on every stop; and
    click thresholds were usually pushed to extremes to fight jitter in an
    unsmoothed pinch measurement that no longer exists. The region is kept but
    flagged, because it was calibrated for the old fingertip tracking point.
    """
    if raw.get("version", 1) >= VERSION:
        return raw
    raw = copy.deepcopy(raw)
    raw.pop("camera", None)
    pointer = raw.setdefault("pointer", {})
    for key in ("min_cutoff", "beta", "lead_ms", "pinch_on", "pinch_off",
                "right_pinch_on", "right_pinch_off"):
        pointer.pop(key, None)
    pointer["needs_calibration"] = True
    raw["version"] = VERSION
    return raw


def load(path: str | None = None) -> dict:
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _merge(DEFAULTS, _migrate(json.load(handle)))
    except (OSError, ValueError):
        # A corrupt config should cost the customisation, not the session.
        return copy.deepcopy(DEFAULTS)


def save(config: dict, path: str | None = None) -> None:
    path = path or CONFIG_PATH
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    os.replace(tmp, path)

"""Settings, with defaults that are meant to be usable before you touch anything.

Stored next to the project as ``config.json``. Unknown keys in the file are kept
and missing ones are filled from the defaults, so an old config keeps working
after new settings are added.
"""

from __future__ import annotations

import copy
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config.json")

DEFAULTS: dict = {
    "transport": "usb",
    "wifi_host": "",
    "port": 9871,
    "autoconnect": True,

    "camera": {
        "fps": 120,
        "shutter": 500,
        "iso_auto": True,
        "iso": 800,
        "iso_ceiling": 4000,
        "zoom": 1.0,
        "focus_auto": True,
        "lens": 0.55,
        "orientation": -1,
    },

    "mode": "off",

    # Which part of the camera's view maps onto the screen, as
    # (left, top, right, bottom) in normalised image coordinates. The default is
    # a middle box rather than the whole frame: reaching the physical corners of
    # the camera's field of view means fully extending your arm, and the corners
    # are also where the lens distorts most.
    "pointer": {
        "region": [0.18, 0.15, 0.82, 0.85],
        # The camera faces you, so it sees you the way another person would:
        # your right hand appears on its left. Without this the cursor goes the
        # wrong way and nothing else can be judged.
        "flip_x": True,
        "flip_y": False,
        "min_cutoff": 1.2,
        "beta": 0.03,
        "lead_ms": 0.0,
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


def load(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _merge(DEFAULTS, json.load(handle))
    except (OSError, json.JSONDecodeError):
        # A corrupt config should cost the customisation, not the session.
        return copy.deepcopy(DEFAULTS)


def save(config: dict, path: str = CONFIG_PATH) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    os.replace(tmp, path)

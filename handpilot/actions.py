"""Everything a gesture can be bound to.

Ordering inside a category is by how likely it is to be wanted, because this list
is rendered into a dropdown that gets opened fifteen times while setting up five
gestures.

Actions take a ``ctx`` so a few of them can reach back into the running app --
switching modes by gesture is the difference between a system you can drive and
one you have to keep going back to the keyboard for.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable

from . import winput


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    category: str
    run: Callable
    #: True if the action reads its parameter box (hotkey text, command line).
    takes_param: bool = False


def _tap(key: str):
    return lambda ctx, param: winput.tap(key)


def _combo(spec: str):
    return lambda ctx, param: winput.hotkey(spec)


def _mouse(fn, *args):
    return lambda ctx, param: fn(*args)


def _custom_hotkey(ctx, param):
    if param:
        winput.hotkey(param)


def _run_command(ctx, param):
    if not param:
        return
    # Detached and without a console window: a gesture that flashes a black box
    # on every trigger is worse than no gesture.
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        subprocess.Popen(param, shell=True, creationflags=flags,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, cwd=os.path.expanduser("~"))
    except OSError as exc:
        ctx.log(f"command failed: {exc}")


_ACTION_LIST: list[Action] = [
    Action("none", "— nothing —", "None", lambda ctx, param: None),

    Action("volume_up", "Volume up", "Audio", _tap("volumeup")),
    Action("volume_down", "Volume down", "Audio", _tap("volumedown")),
    Action("volume_mute", "Mute / unmute", "Audio", _tap("volumemute")),

    Action("media_play_pause", "Play / pause", "Media", _tap("playpause")),
    Action("media_next", "Next track", "Media", _tap("nexttrack")),
    Action("media_prev", "Previous track", "Media", _tap("prevtrack")),
    Action("media_stop", "Stop playback", "Media", _tap("stop")),

    Action("click_left", "Left click", "Mouse", _mouse(winput.click, "left")),
    Action("click_right", "Right click", "Mouse", _mouse(winput.click, "right")),
    Action("click_double", "Double click", "Mouse", _mouse(winput.double_click, "left")),
    Action("click_middle", "Middle click", "Mouse", _mouse(winput.click, "middle")),
    Action("scroll_up", "Scroll up", "Mouse", _mouse(winput.wheel, 3)),
    Action("scroll_down", "Scroll down", "Mouse", _mouse(winput.wheel, -3)),
    Action("scroll_left", "Scroll left", "Mouse", _mouse(winput.hwheel, -3)),
    Action("scroll_right", "Scroll right", "Mouse", _mouse(winput.hwheel, 3)),

    Action("slide_next", "Next slide", "Presentation", _tap("right")),
    Action("slide_prev", "Previous slide", "Presentation", _tap("left")),
    Action("slide_start", "Start slideshow", "Presentation", _tap("f5")),
    Action("slide_end", "End slideshow", "Presentation", _tap("escape")),
    Action("page_down", "Page down", "Presentation", _tap("pagedown")),
    Action("page_up", "Page up", "Presentation", _tap("pageup")),

    Action("key_enter", "Enter", "Keys", _tap("enter")),
    Action("key_escape", "Escape", "Keys", _tap("escape")),
    Action("key_space", "Space", "Keys", _tap("space")),
    Action("key_tab", "Tab", "Keys", _tap("tab")),
    Action("key_delete", "Delete", "Keys", _tap("delete")),
    Action("key_left", "Arrow left", "Keys", _tap("left")),
    Action("key_right", "Arrow right", "Keys", _tap("right")),
    Action("key_up", "Arrow up", "Keys", _tap("up")),
    Action("key_down", "Arrow down", "Keys", _tap("down")),

    Action("alt_tab", "Switch window (Alt+Tab)", "Windows", _combo("alt+tab")),
    Action("task_view", "Task view", "Windows", _combo("win+tab")),
    Action("show_desktop", "Show desktop", "Windows", _combo("win+d")),
    Action("snap_left", "Snap window left", "Windows", _combo("win+left")),
    Action("snap_right", "Snap window right", "Windows", _combo("win+right")),
    Action("minimize_window", "Minimise window", "Windows", _combo("win+down")),
    Action("maximize_window", "Maximise window", "Windows", _combo("win+up")),
    Action("close_window", "Close window", "Windows", _combo("alt+f4")),
    Action("desktop_left", "Desktop left", "Windows", _combo("ctrl+win+left")),
    Action("desktop_right", "Desktop right", "Windows", _combo("ctrl+win+right")),
    Action("lock_pc", "Lock the PC", "Windows", _combo("win+l")),
    Action("screenshot_region", "Screenshot a region", "Windows", _combo("win+shift+s")),

    Action("browser_back", "Back", "Browser", _tap("browserback")),
    Action("browser_forward", "Forward", "Browser", _tap("browserforward")),
    Action("browser_refresh", "Refresh", "Browser", _tap("browserrefresh")),
    Action("new_tab", "New tab", "Browser", _combo("ctrl+t")),
    Action("close_tab", "Close tab", "Browser", _combo("ctrl+w")),
    Action("reopen_tab", "Reopen closed tab", "Browser", _combo("ctrl+shift+t")),
    Action("next_tab", "Next tab", "Browser", _combo("ctrl+tab")),

    Action("copy", "Copy", "Editing", _combo("ctrl+c")),
    Action("paste", "Paste", "Editing", _combo("ctrl+v")),
    Action("cut", "Cut", "Editing", _combo("ctrl+x")),
    Action("undo", "Undo", "Editing", _combo("ctrl+z")),
    Action("redo", "Redo", "Editing", _combo("ctrl+y")),
    Action("save", "Save", "Editing", _combo("ctrl+s")),
    Action("select_all", "Select all", "Editing", _combo("ctrl+a")),
    Action("find", "Find", "Editing", _combo("ctrl+f")),
    Action("zoom_in", "Zoom in", "Editing", _combo("ctrl+plus")),
    Action("zoom_out", "Zoom out", "Editing", _combo("ctrl+minus")),
    Action("zoom_reset", "Reset zoom", "Editing", _combo("ctrl+0")),

    Action("mode_pointer", "Switch to pointer mode", "HandPilot",
           lambda ctx, param: ctx.set_mode("pointer")),
    Action("mode_gestures", "Switch to gesture mode", "HandPilot",
           lambda ctx, param: ctx.set_mode("gestures")),
    Action("mode_cycle", "Cycle mode", "HandPilot", lambda ctx, param: ctx.cycle_mode()),
    Action("mode_off", "Stop controlling", "HandPilot", lambda ctx, param: ctx.set_mode("off")),

    Action("custom_hotkey", "Custom hotkey…", "Custom", _custom_hotkey, takes_param=True),
    Action("run_command", "Run a command…", "Custom", _run_command, takes_param=True),
]

ACTIONS: dict[str, Action] = {a.id: a for a in _ACTION_LIST}


def ordered_ids() -> list[str]:
    return [a.id for a in _ACTION_LIST]


def label_for(action_id: str) -> str:
    action = ACTIONS.get(action_id)
    if action is None:
        return action_id
    return action.label if action.category == "None" else f"{action.category}: {action.label}"


def id_for_label(label: str) -> str:
    for action in _ACTION_LIST:
        if label_for(action.id) == label:
            return action.id
    return "none"


def run(action_id: str, ctx, param: str = "") -> None:
    action = ACTIONS.get(action_id)
    if action is None or action.id == "none":
        return
    try:
        action.run(ctx, param)
    except Exception as exc:                              # noqa: BLE001
        # A bad hotkey string or a missing key should cost that one trigger, not
        # the control loop that is holding the mouse button down.
        ctx.log(f"action {action_id} failed: {exc}")

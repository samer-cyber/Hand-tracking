"""Synthetic mouse and keyboard input, via ``SendInput``.

``SendInput`` rather than ``SetCursorPos`` and ``keybd_event``: it goes through
the same path real hardware does, so applications that track mouse *movement*
(games, drawing tools, anything using raw input or hooks) see a moving pointer
rather than one that teleports. The difference shows up immediately in anything
that cares about drag.

ctypes rather than pyautogui or pynput, so running this is `python run.py` with
nothing to install first.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000
MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x8000, 0x4000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_EXTENDEDKEY = 0x0001

SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send(*inputs: INPUT) -> None:
    count = len(inputs)
    array = (INPUT * count)(*inputs)
    user32.SendInput(count, ctypes.byref(array), ctypes.sizeof(INPUT))


def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    return INPUT(type=INPUT_MOUSE,
                 mi=MOUSEINPUT(dx=dx, dy=dy, mouseData=data & 0xFFFFFFFF,
                               dwFlags=flags, time=0, dwExtraInfo=0))


def _key(vk: int, up: bool) -> INPUT:
    flags = KEYEVENTF_KEYUP if up else 0
    # Extended-key flag for the keys that live on the duplicated block. Without
    # it some applications read Right-arrow as the numpad 6 they share a scan
    # code with, which is exactly the key most likely to be bound to "next slide".
    if vk in _EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    return INPUT(type=INPUT_KEYBOARD,
                 ki=KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0))


def set_dpi_aware() -> None:
    """Without this Windows reports a virtual desktop scaled by the display's DPI
    setting, and the pointer lands short of the real screen edges by exactly the
    scaling factor."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def virtual_screen() -> tuple[int, int, int, int]:
    """(left, top, width, height) across every monitor."""
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def cursor_position() -> tuple[int, int]:
    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def move_to(x: int, y: int) -> None:
    """Absolute move in virtual-desktop pixels."""
    left, top, width, height = virtual_screen()
    if width <= 1 or height <= 1:
        return
    nx = int(round((x - left) * 65535 / (width - 1)))
    ny = int(round((y - top) * 65535 / (height - 1)))
    nx = min(65535, max(0, nx))
    ny = min(65535, max(0, ny))
    _send(_mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, nx, ny))


_BUTTONS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


def button_down(button: str = "left") -> None:
    _send(_mouse(_BUTTONS[button][0]))


def button_up(button: str = "left") -> None:
    _send(_mouse(_BUTTONS[button][1]))


def click(button: str = "left") -> None:
    down, up = _BUTTONS[button]
    _send(_mouse(down), _mouse(up))


def double_click(button: str = "left") -> None:
    down, up = _BUTTONS[button]
    _send(_mouse(down), _mouse(up))
    # Inside the double-click interval but not so tight that the target window
    # coalesces the pair into one press.
    time.sleep(0.03)
    _send(_mouse(down), _mouse(up))


def wheel(clicks: float) -> None:
    _send(_mouse(MOUSEEVENTF_WHEEL, data=int(round(clicks * 120))))


def hwheel(clicks: float) -> None:
    _send(_mouse(MOUSEEVENTF_HWHEEL, data=int(round(clicks * 120))))


VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "shift": 0x10, "ctrl": 0x11,
    "alt": 0x12, "pause": 0x13, "capslock": 0x14, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E,
    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C, "apps": 0x5D,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stop": 0xB2, "playpause": 0xB3,
    "browserback": 0xA6, "browserforward": 0xA7, "browserrefresh": 0xA8,
    "plus": 0xBB, "minus": 0xBD, "0": 0x30,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())
for _d in "0123456789":
    VK[_d] = ord(_d)

_EXTENDED = {
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E,
    0x5B, 0x5C, 0x5D, 0xA6, 0xA7, 0xA8, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
}

_MODIFIERS = {"ctrl", "control", "alt", "shift", "win", "cmd", "meta"}
_MOD_VK = {"ctrl": 0x11, "control": 0x11, "alt": 0x12, "shift": 0x10,
           "win": 0x5B, "cmd": 0x5B, "meta": 0x5B}


def tap(key: str) -> None:
    vk = VK.get(key.lower())
    if vk is None:
        return
    _send(_key(vk, False), _key(vk, True))


def hotkey(spec: str) -> bool:
    """Press a combination written as ``ctrl+shift+p``. Returns False if any part
    of it is not a key this module knows."""
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return False
    mods = [p for p in parts if p in _MODIFIERS]
    keys = [p for p in parts if p not in _MODIFIERS]
    vks = [VK.get(k) for k in keys]
    if any(v is None for v in vks):
        return False

    events = [_key(_MOD_VK[m], False) for m in mods]
    events += [_key(v, False) for v in vks]
    events += [_key(v, True) for v in reversed(vks)]
    events += [_key(_MOD_VK[m], True) for m in reversed(mods)]
    _send(*events)
    return True


def release_all_buttons() -> None:
    """Panic path. Any code that puts a button down has to be able to guarantee it
    comes back up, or a lost connection leaves the desktop with the left button
    held and the machine effectively unusable."""
    for button in _BUTTONS:
        try:
            button_up(button)
        except OSError:
            pass

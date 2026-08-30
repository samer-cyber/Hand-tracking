"""Windows cursor control via SendInput.

Uses the Win32 SendInput API directly through ctypes rather than a convenience
wrapper such as pyautogui. Wrappers typically add a per-call sleep and go
through several layers of Python before reaching the OS, which costs
milliseconds on every single cursor update -- and in a system whose central
problem is latency, spending them here would be careless.

Safety matters in this module more than anywhere else in the project. A tracker
that owns the cursor and cannot be switched off is genuinely difficult to
recover from: you cannot reliably click the button that disables it. So control
is off until explicitly enabled, movement stops the moment tracking is lost, and
the caller is expected to provide a keyboard escape that does not require
pointing at anything.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTunion)]


def enable_dpi_awareness() -> None:
    """Report true pixel coordinates on scaled displays.

    Without this, a display at 125% or 150% scaling reports a smaller logical
    resolution than it physically has, and the cursor lands consistently short
    of where it should -- an error that looks exactly like a calibration bug
    and wastes a lot of time to find.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)     # per-monitor aware
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def _send(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
    inp = INPUT(type=INPUT_MOUSE,
                union=_INPUTunion(mi=MOUSEINPUT(dx, dy, data, flags, 0, None)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


VK_ESCAPE, VK_F9, VK_F10 = 0x1B, 0x78, 0x79


def key_held(vk: int) -> bool:
    """True while a key is physically down, regardless of window focus.

    This is the escape hatch, and it is not a nicety. Once the tracker is moving
    the cursor you may not be able to click the preview window to give it focus,
    so a key handler that only fires when that window is focused can leave you
    with no way to stop it. GetAsyncKeyState reads the hardware state directly
    and works no matter what is in front.
    """
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


class _Edge:
    """Fires once per press rather than continuously while held."""

    def __init__(self, vk: int):
        self.vk = vk
        self._was = False

    def pressed(self) -> bool:
        now = key_held(self.vk)
        fired = now and not self._was
        self._was = now
        return fired


class VirtualScreen:
    """Geometry of the whole desktop, including multiple monitors."""

    def __init__(self):
        enable_dpi_awareness()
        self.x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        self.y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        self.width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
        self.height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    def to_absolute(self, px: float, py: float) -> tuple[int, int]:
        """Screen pixels to the 0-65535 normalised space SendInput expects."""
        nx = (px - self.x) / max(self.width - 1, 1)
        ny = (py - self.y) / max(self.height - 1, 1)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        return int(round(nx * 65535)), int(round(ny * 65535))

    def __repr__(self) -> str:
        return f"VirtualScreen({self.width}x{self.height} at {self.x},{self.y})"


class MouseController:
    """Cursor and button control, disabled until explicitly switched on."""

    def __init__(self, screen: VirtualScreen | None = None):
        self.screen = screen or VirtualScreen()
        self.enabled = False
        self.clicks_enabled = False
        self.left_down = False
        self._last: tuple[int, int] | None = None

    def move_to(self, px: float, py: float) -> None:
        if not self.enabled:
            return
        ax, ay = self.screen.to_absolute(px, py)
        if self._last == (ax, ay):
            return                      # skip no-op events; they still cost a syscall
        self._last = (ax, ay)
        _send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, ax, ay)

    def press_left(self) -> None:
        if not (self.enabled and self.clicks_enabled) or self.left_down:
            return
        self.left_down = True
        _send(MOUSEEVENTF_LEFTDOWN)

    def release_left(self) -> None:
        if not self.left_down:
            return
        self.left_down = False
        _send(MOUSEEVENTF_LEFTUP)

    def click_right(self) -> None:
        if not (self.enabled and self.clicks_enabled):
            return
        _send(MOUSEEVENTF_RIGHTDOWN)
        _send(MOUSEEVENTF_RIGHTUP)

    def scroll(self, amount: int) -> None:
        if not self.enabled:
            return
        _send(MOUSEEVENTF_WHEEL, data=amount)

    def release_all(self) -> None:
        """Always call this on shutdown: a stuck mouse button is miserable."""
        if self.left_down:
            self.release_left()

    def set_enabled(self, on: bool) -> None:
        if not on:
            self.release_all()
        self.enabled = on
        self._last = None

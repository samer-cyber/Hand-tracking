"""Interrupt handling and guaranteed cleanup for the interactive scripts.

Two problems this solves.

First, Ctrl+C is unreliable in an OpenCV GUI loop. Python only delivers SIGINT
between bytecode instructions, so while the interpreter is inside a long C call
the signal simply waits. Installing an explicit handler that sets a flag, and
testing that flag once per frame, makes interruption deterministic rather than
dependent on where the interpreter happened to be.

Second, these scripts hold resources whose leak is actively harmful -- camera
handles, and on the mouse path a possibly-held mouse button. Cleanup therefore
has to run on *every* exit path, including exceptions and interrupts, not just
the one where the loop ends normally.
"""
from __future__ import annotations

import signal
import sys
import traceback


class Interrupt:
    """Cooperative stop flag fed by SIGINT.

    A second Ctrl+C exits immediately, on the assumption that if the graceful
    path had been going to work it would have worked the first time.
    """

    def __init__(self):
        self.requested = False
        self._count = 0
        self._previous = None

    def install(self) -> "Interrupt":
        def handler(signum, frame):
            self._count += 1
            self.requested = True
            if self._count == 1:
                print("\ninterrupt received -- shutting down cleanly "
                      "(press Ctrl+C again to force)", flush=True)
            else:
                print("\nforced exit", flush=True)
                sys.exit(130)
        try:
            self._previous = signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):        # not on the main thread
            pass
        return self

    def __bool__(self) -> bool:
        return self.requested


def run_guarded(body, cleanup) -> int:
    """Run `body`, and run `cleanup` no matter how it ends.

    Returns a process exit code. Interrupts are not treated as failures; an
    unexpected exception is reported with a traceback but cleanup still runs,
    because leaving a camera locked or a mouse button held would turn one bug
    into a broken machine.
    """
    code = 0
    try:
        code = body() or 0
    except KeyboardInterrupt:
        print("\ninterrupted")
        code = 130
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        try:
            cleanup()
        except Exception:
            print("cleanup failed:")
            traceback.print_exc()
    return code

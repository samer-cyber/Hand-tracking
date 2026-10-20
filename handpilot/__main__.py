"""Command line.

    python run.py                    open HandPilot: the window, and mouse control
    python run.py --stream           raw hand data as JSON lines -- no window, no mouse
    python run.py --stream --csv --out hand.csv --seconds 30
    python run.py --doctor           check the connection to the phone, step by step
    python run.py --no-window        run without opening a window (prints the address)

``python -m handpilot`` takes the same arguments.
"""

from __future__ import annotations

import argparse
import sys
import threading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="handpilot",
        description="Your iPhone as a 120 fps hand tracker: drive the mouse, or just read the data.")
    parser.add_argument("--stream", action="store_true",
                        help="print raw landmarks instead of opening the app (no mouse control)")
    parser.add_argument("--csv", action="store_true", help="with --stream: CSV instead of JSON lines")
    parser.add_argument("--out", default="", metavar="FILE", help="with --stream: write to FILE")
    parser.add_argument("--seconds", type=float, default=None,
                        help="with --stream: stop after this many seconds")
    parser.add_argument("--wifi", default="", metavar="IP",
                        help="with --stream/--doctor: reach the phone over Wi-Fi instead of USB")
    parser.add_argument("--port", type=int, default=9871, help="the port set in the iPhone app")
    parser.add_argument("--doctor", action="store_true", help="diagnose the connection and exit")
    parser.add_argument("--no-window", action="store_true", help="don't open the app window")
    parser.add_argument("--ui-port", type=int, default=None, help="port for the app window")
    args = parser.parse_args(argv)

    if args.doctor:
        from .doctor import main as doctor
        return doctor(port=args.port, wifi=args.wifi)

    if args.stream:
        from .stream import write_stream
        out = open(args.out, "w", encoding="utf-8", newline="") if args.out else sys.stdout
        if args.out:
            print(f"recording to {args.out} -- Ctrl+C to stop", file=sys.stderr)
        try:
            count = write_stream(out, "csv" if args.csv else "json", wifi=args.wifi,
                                 port=args.port, seconds=args.seconds)
        finally:
            if args.out:
                out.close()
        print(f"{count} frames", file=sys.stderr)
        return 0

    if sys.platform != "win32":
        print("Mouse control is Windows-only for now. The raw data works everywhere:\n"
              "    python run.py --stream", file=sys.stderr)
        return 2
    return _run_app(args)


def _run_app(args) -> int:
    from . import config, winput
    from .app import App
    from .server import UIServer, ping_existing

    winput.set_dpi_aware()
    cfg = config.load()
    app = App(cfg)
    port = args.ui_port or int(cfg.get("ui", {}).get("port", 9870))
    try:
        server = UIServer(app, port, quit_on_close=not args.no_window)
    except OSError:
        if ping_existing(port):
            print("HandPilot is already running; its window has been brought up.")
            return 0
        server = UIServer(app, 0, quit_on_close=not args.no_window)

    app.start()
    threading.Thread(target=server.serve_forever, name="handpilot-ui", daemon=True).start()
    print("HandPilot is running.")
    print(f"  window: {server.url}")
    print("  F9 releases control from anywhere. Close the window or press Ctrl+C to quit.")
    if not args.no_window:
        server.open_window()
    try:
        while not app.quit_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

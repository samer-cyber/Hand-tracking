# Contributing

Bug reports, fixes and ideas are welcome.

## Reporting a problem

Open an issue and include:

- what you expected and what happened;
- the output of `python run.py --doctor` for anything connection-related;
- your iPhone model and iOS version, and your Windows version.

## Working on the desktop side

Python 3.10+, standard library only, so there is nothing to install. Please keep it that way: "clone
and run" is a feature.

```
python tools/selftest.py
```

runs the desktop half against synthesised hands and a fake phone on a socket, with the mouse stubbed
out and no hardware needed. Run it before and after any change. If you change a threshold or the
filter, add a check that pins down the behaviour you meant.

| | |
|---|---|
| `handpilot/usbmux.py` | Apple's USB multiplexer protocol |
| `handpilot/link.py`, `protocol.py` | connect, read, reconnect, parse the wire format |
| `handpilot/handmodel.py` | landmarks to aspect-corrected, palm-normalised features |
| `handpilot/filters.py` | One Euro filter, dead zone, hysteresis, the steadiness curve |
| `handpilot/controller.py` | pointer mode, calibration, gestures, hotkeys |
| `handpilot/stream.py` | the raw-data API, UDP output, `--stream` |
| `handpilot/app.py`, `server.py`, `web/` | the window: settings schema, local server, page |
| `handpilot/winput.py`, `actions.py`, `gestures.py` | Windows input, gesture actions, poses |

## Working on the iPhone app

The app is SwiftUI and AVFoundation, in `ios/HandPilot/HandPilot/`, with the project described by
`ios/HandPilot/project.yml` (XcodeGen). Without a Mac, check that your change compiles by running the
Codemagic or GitHub Actions build (see [docs/ios-build.md](docs/ios-build.md)) before opening a pull
request. Keep the deployment target at iOS 17.

If you change the wire format, change `Wire.swift` and `handpilot/protocol.py` together, and extend
the wire tests in `tools/selftest.py`.

## Style

Match the code around you. Comments explain *why*, especially the measured reasons behind numbers;
the code already says *what*.

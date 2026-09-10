# HandPilot

Your iPhone's rear camera, running Apple's hand-pose model on the Neural Engine,
driving your Windows cursor and a set of gestures over the USB cable.

**Setup instructions are in [SETUP.md](SETUP.md).** This file is what it is and
why it is built this way.

---

## The shape of it

```
iPhone 17, rear wide camera            Windows
┌──────────────────────────────┐      ┌────────────────────────────┐
│ 1080p @ 120 fps, 1/500 s     │      │ usbmux client              │
│          ↓                   │      │      ↓                     │
│ VNDetectHumanHandPoseRequest │      │ features → pose / pointer  │
│  (Neural Engine, on device)  │      │      ↓                     │
│          ↓                   │      │ SendInput                  │
│ 21 landmarks → 291 bytes     │─USB─▶│                            │
│ TCP server, port 9871        │      │                            │
└──────────────────────────────┘      └────────────────────────────┘
```

No video ever leaves the phone. Each frame becomes 291 bytes — a header plus
21 × (x, y, confidence) — which is about 35 KB/s at 120 fps and nothing at all to
a USB cable.

### Why the phone is the server

Apple's USB multiplexer can open a tunnel from the PC *into* a TCP port on the
phone. There is no route the other way. So the phone listens and the PC dials in,
which is backwards from how you would write this over Wi-Fi, and is the single
structural decision the rest of the protocol follows from.

`handpilot/usbmux.py` speaks that protocol directly — the same thing `iproxy` and
Xcode do — so there is no helper binary to install and no port-forwarding process
to leave running. The same phone-side listener also serves plain Wi-Fi with no
code change; the desktop just dials an IP instead.

## Where the latency actually goes

The interesting number is not any single stage, it is which stage you are
*allowed* to fix. Roughly, per frame:

| stage | cost | notes |
|---|---|---|
| exposure | 2 ms | 1/500 s. This is the motion-blur term, not really latency |
| sensor readout and delivery | ~8 ms | one frame period at 120 fps — the floor |
| Vision hand pose | measured live | shown in the app; expect single digits |
| USB round trip | measured live | typically well under a millisecond |
| desktop maths | <0.3 ms | shown in the app |
| Windows input → pixels on screen | not measurable from here | compositor and display |

The app measures and displays every term it can, so "it feels laggy" turns into a
number rather than a debate. Expect somewhere in the high teens of milliseconds
from hand movement to cursor movement, most of it in the two stages nobody can
remove: the frame period and the display.

Two settings exist specifically to buy back what is left. **Lead ahead** on the
Pointer tab extrapolates along the current velocity, which trades a little
overshoot on direction changes for a cursor that arrives when your hand does.
**Responsiveness** opens the smoothing filter up as your hand speeds up.

## The two decisions that matter most on the camera

**Frame rate is a latency floor.** At 30 fps you are on average 16 ms out of date
about where the hand is before any processing has started, and no amount of
downstream cleverness recovers that. 120 fps cuts it to 4 ms.

**Shutter speed is an accuracy floor.** Frame rate bounds how *often* you learn
where the hand is; shutter speed bounds how *smeared* each of those measurements
is. A hand crossing the frame in 300 ms moves several of its own widths during a
1/60 s exposure, and the landmark model then fits fingers to a streak. 1/500 s
removes that — at the cost of roughly three stops of light, which is why there is
an ISO servo and why a dim room is the one thing that will genuinely hold this
setup back.

## Files

**Phone** — `ios/HandPilot/HandPilot/`

| | |
|---|---|
| `CaptureEngine.swift` | camera format, manual exposure, Vision, per-frame packet |
| `Server.swift` | TCP listener, backpressure, JSON command channel |
| `Wire.swift` | the packet format |
| `ContentView.swift` | on-phone controls and status |

**Desktop** — `handpilot/`

| | |
|---|---|
| `usbmux.py` | Apple's USB multiplexer protocol |
| `link.py` | connect, read, reconnect, measure round trip |
| `protocol.py` | packet parsing, resynchronising |
| `handmodel.py` | landmarks → aspect-corrected, palm-normalised features |
| `gestures.py` | pose classification, swipes, hold/repeat/cooldown |
| `filters.py` | One Euro filter, hysteresis latches |
| `controller.py` | pointer mode, gesture mode, global hotkeys |
| `winput.py` | `SendInput` mouse and keyboard |
| `actions.py` | the ~60 things a gesture can be bound to |
| `ui.py` | the window |

**Tools**

```bash
python tools/selftest.py
```

Runs the whole desktop half against synthesised hands, poses and a fake phone on
a socket — 56 checks, no hardware needed. Worth running after changing any
threshold, because the alternative way to discover a bad one is to stand in front
of a camera waving.

```bash
python run.py --doctor
```

Walks the connection path one link at a time and names the specific fix, because
"no Apple service", "no cable", "phone locked" and "app not started" all present
identically as *it does not connect*.

## Two things worth knowing before you use it

**The phone gets warm.** 1080p120 plus continuous neural-engine inference is a
real thermal load, and it is charging over the same cable. The status panel shows
the thermal state; if it reaches *serious*, drop to 60 fps or turn off the phone's
preview.

**Pointer mode can run away with your mouse.** **F9** releases control from
anywhere, and it is registered as a global hotkey precisely because the failure
mode of a hand-driven cursor is one where clicking a Stop button is the thing you
cannot reliably do.

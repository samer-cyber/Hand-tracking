# How it works

```
iPhone                                    Computer
┌────────────────────────────────┐       ┌──────────────────────────────────┐
│ camera, 1080p @ 120 fps,       │       │ usbmux client                    │
│ shutter fixed at 1/500 s       │       │        ↓                         │
│        ↓                       │       │ landmarks → features             │
│ Apple Vision hand pose         │       │        ↓                         │
│ (Neural Engine, on the phone)  │       │ filter → cursor   or  raw data   │
│        ↓                       │       │        ↓                ↓        │
│ 21 landmarks → 291 bytes       │─USB──▶│ SendInput      Python/UDP/web    │
│ TCP server, port 9871          │       │                                  │
└────────────────────────────────┘       └──────────────────────────────────┘
```

No video leaves the phone. Each frame becomes 291 bytes: a header plus 21 × (x, y, confidence).
That's about 35 KB/s at 120 fps, nothing at all to a USB cable.

## Why the phone is the server

Apple's USB multiplexer (usbmuxd, the thing iTunes and Xcode use) can open a tunnel from a computer
*into* a TCP port on the phone, but there's no route the other way. So the phone listens and the
computer dials in, the opposite of how you'd write this over Wi-Fi. That is the single structural
decision the rest of the protocol follows from.

[`handpilot/usbmux.py`](../handpilot/usbmux.py) speaks that protocol directly, the same thing
`iproxy` and Xcode do. So there's no helper binary to install and no port-forwarding process left
running. The same listener serves Wi-Fi with no change: the computer just dials an IP instead.

## The two camera decisions that matter most

**Frame rate is a latency floor.** At 30 fps you're on average 16 ms out of date about where the hand
is before any processing starts, and nothing downstream can recover that. At 120 fps it's 4 ms.

**Shutter speed is an accuracy floor.** Frame rate bounds how *often* you learn where the hand is;
shutter speed bounds how *smeared* each measurement is. A hand crossing the frame in 300 ms moves
several of its own widths during a 1/60 s exposure, and the landmark model then fits fingers to a
streak. 1/500 s removes that, at the cost of roughly three stops of light. That's why the app runs an
ISO servo, and why a dim room is the one thing that holds this setup back.

Measured with MediaPipe on synthetic images: landmark error grows only slightly with blur up to about
10% of the palm's length, then climbs. Past about 40%, landmarks drift by up to a third of a palm, and
past 60% the hand is often not found at all. A hand moving 1 m/s smears about 37% of a palm in a
typical webcam's 1/30 s exposure, and about 2% at 1/500 s.

## Where the latency goes

| Stage | Cost | Notes |
|---|---|---|
| exposure | 2 ms | 1/500 s; this is the blur term more than delay |
| sensor readout and delivery | ~8 ms | one frame period at 120 fps: the floor |
| Vision hand pose | measured live | shown in the window; single-digit milliseconds on recent iPhones |
| USB round trip | measured live | typically well under a millisecond |
| desktop maths | < 0.3 ms | measured live |
| Windows input → pixels | not measurable from here | compositor and display |

The window shows every term it can measure, so "it feels laggy" turns into a number rather than a
debate. Most of the total sits in the two stages nobody can remove: the frame period and the display.

## From landmarks to a cursor

Per frame, on the thread that reads the socket (no queue, no scheduler hop):

1. **Tracking point.** The average of the four knuckles, weighted by confidence. Fingertips are the
   noisiest points a hand model reports, and the thumb tip is the worst. The knuckles barely move when
   you pinch, which is exactly when the cursor must not move.
2. **Screen area.** The calibrated part of the camera's view is stretched over the main monitor,
   mirrored so your right hand moves the cursor right.
3. **One Euro filter** ([Casiez et al., CHI 2012](https://gery.casiez.net/1euro/)). Heavy smoothing
   while the hand is still, almost none while it moves. It runs on the phone's capture timestamps,
   which are exactly one frame apart. The PC's arrival times come in bursts, because frames from one
   USB read land together, and a speed-sensitive filter is only as good as its clock.
4. **Rope dead zone.** The cursor only moves once the filtered point pulls more than a pixel or two
   away, then follows at that distance. A still hand holds the cursor perfectly still; a moving hand
   loses a pixel or two of lag, which nobody can see.
5. **Clicks** come from the thumb-to-index gap, measured in palm widths so it doesn't matter how far
   away you sit, with separate press and release thresholds. When a pinch closes, the cursor freezes
   at where it was a moment before the fingers started moving. Move further than the freeze radius
   and it becomes a drag.

One **Steadiness** control sets the filter and the dead zone together, along a curve that stays near
the best jitter-for-lag trade-off.

## How the jitter was fixed

The first version shook visibly. Four separate causes stacked up:

- The **Smoothing** slider ran backwards. It set the filter's cutoff frequency, so dragging it up meant
  *less* smoothing. Users naturally pushed it to maximum.
- **Lead ahead**, an extrapolation along the current velocity, overshot by tens of pixels every time
  the hand stopped, then settled over half a second. On screen that looks like vibration.
- The cursor rode on the **thumb tip**, the noisiest landmark.
- The filter ran on **arrival time**, not capture time.

The same pipeline with simulated landmark noise, before and after:

| | Still hand | Lag, slow / fast move | When you stop |
|---|---|---|---|
| Before, with the settings a user had ended up with | shakes ±2.2 px, moves 315× in 3 s | runs 9–12 ms *ahead* | overshoots up to 27 px, ~0.75 s to settle |
| Before, shipped defaults | shakes ±1.3 px, moves 154× in 3 s | 7 / 3 ms | still jittering 0.2–0.5 s later |
| Now, *Balanced* | dead still | 15 / 7 ms | no overshoot, settled in ~40 ms |

The last piece is on the phone. Apple Vision detects the hand afresh in every frame rather than
tracking it. Measured with MediaPipe, that kind of per-frame detection jitters about twice as much as
tracking, and up to six times as much when the hand is small in the frame. **Hand crop** gives the model
a crop around where the hand just was, so the hand fills more of the model's input, like optical zoom
without losing the wide view needed to find it again. The app checks once, on a real frame, how Vision
reports coordinates for a crop, so a wrong assumption can't misplace the hand.

## The window

A local web page rather than a native toolkit: every Windows machine already has a modern renderer,
and nothing needs installing. `python run.py` serves it on `127.0.0.1` and opens it in an Edge or Chrome
app window. The control loop never waits for it. The page reads a snapshot 30 times a second over
server-sent events; measured, that costs the tracking thread nothing.

Since this server can move your mouse, it only listens on 127.0.0.1. Every API call needs a token
generated for that run, and requests whose Host header isn't 127.0.0.1 or localhost are refused,
which blocks DNS-rebinding tricks.

## Why no dependencies

The desktop side uses only Python's standard library: the window is served with `http.server`, input
goes through `ctypes`, and the USB tunnel uses `plistlib` and `socket`. A dependency would mean a pip
step between wanting to try it and trying it, and none of the candidates would have earned that.

## Heat

1080p at 120 fps plus continuous inference on the Neural Engine is a real thermal load, often while
charging over the same cable. The iPhone reading at the bottom of the window shows the phone's thermal state. If it reaches *Hot*, drop
to 60 fps or switch the phone's screen to *Skeleton*, which is cheaper than the live camera picture.

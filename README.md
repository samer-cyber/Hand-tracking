# HandPilot

**Your iPhone, turned into a 120 fps hand tracker for your computer.**
Steer the mouse with your hand, or just read the 21 hand landmarks from your own code.

![The HandPilot desktop app, tracking a (simulated) hand](docs/images/desktop.png)

The iPhone runs Apple's hand-pose model on its Neural Engine, right next to the camera, and sends
only the landmarks to your computer over the USB cable: 291 bytes a frame. No video ever leaves the
phone. Every stage measures itself, so "it feels laggy" becomes a number on screen.

## Why a phone beats a webcam

Webcam hand tracking with MediaPipe works, but it feels smeary and late. The model isn't the
problem: measured on an ordinary laptop, MediaPipe takes about 8–18 ms a frame. **The webcam is.**

| | Laptop webcam + MediaPipe | iPhone + HandPilot |
|---|---|---|
| Frames per second | 30, often 15 in dim light | 60–240 |
| Exposure per frame | ~1/30 s, automatic: a moving hand is a smear | 1/500 s, fixed: a sharp hand |
| Hand moving 1 m/s smears by | ~3 cm, a third of your palm | ~2 mm |
| Image processing | noise reduction that blends frames together | switched off |
| What crosses the cable | video | 21 points |

A smeared hand gives smeared landmarks, and at 30 fps the newest frame is already up to 33 ms old
before any processing starts. A phone camera fixes both, and it's in your pocket already.
[How it works](docs/how-it-works.md) has the latency budget and the details.

## What you can do with it

- **Mouse.** The cursor follows your knuckles. Pinch thumb and index to click, hold the pinch to drag,
  pinch thumb and middle to right-click, and hold two fingers up to scroll. **F9** releases control from
  anywhere, even if the window is hidden.
- **Raw data.** Stream the landmarks into Python, any language over UDP, or a web page. No mouse is
  involved. See [docs/data.md](docs/data.md).
- **Gestures** (experimental, off by default). Poses and swipes trigger actions such as volume or
  next track.

## Quick start

1. **Get the iPhone app.** Build it in the cloud with no Mac (Codemagic or GitHub Actions, about five
   minutes) or with Xcode, then install it with your own Apple ID.
   → [docs/ios-build.md](docs/ios-build.md)
2. **Run the desktop app.** You need Windows and Python 3.10 or newer, and nothing else to install.
   Download this repository (*Code → Download ZIP*, or `git clone`), open a terminal in its folder,
   and run:
   ```
   python run.py
   ```
3. **Connect.** Plug the iPhone in with a cable, open HandPilot on it and tap **Start**. Stand it about
   a forearm's length away with the camera facing you. In the window, click **Calibrate screen area**,
   sweep your hand around, then switch to **Mouse**.

Windows needs Apple's USB driver to see the phone. Install the free **Apple Devices** app from the
Microsoft Store and open it once. Or use Wi-Fi instead (Advanced → Connection).

**Just want the data?** Close the app and:

```python
from handpilot import HandStream

with HandStream() as hands:
    for frame in hands:
        if frame.hand:
            print(frame.landmark("index_tip"))   # (x, y, confidence), x and y in 0..1
```

Or run `python run.py --stream` for JSON lines, or add `--csv --out hand.csv` to record a session.
This part also works on macOS and Linux.

## Using it well

- **Light the room.** A 1/500 s shutter needs light. If the Light tile says *too dark for this
  shutter*, turn on a lamp or choose 1/250.
- **Steadiness** is the one smoothing control: *Responsive*, *Balanced* or *Steady*. The **Stillness**
  tile shows how much the cursor wobbles while your hand is still. Watch it while you try them.
- **Calibrate** to pick the area of the camera's view that becomes your screen. Choose a range you can
  reach without moving your shoulder.

More in the [user guide](docs/using.md), including fixes for anything that doesn't connect.

## Honest limitations

- Pointing in mid-air is less precise than a mouse. Studies put it at roughly 2.8 bits/s (a standard
  speed-and-accuracy score from pointing tests) against 3.7–4.9 for a mouse, and your arm gets tired.
  It's great for presenting, couch control, demos and creative coding. It is not a mouse replacement.
- Mouse control is Windows-only for now.
- Apps sideloaded with a free Apple ID expire after 7 days. Re-sign with Sideloadly; your settings are
  kept.
- Gestures are experimental: poses are easy to trigger by accident.

## Documentation

| | |
|---|---|
| [Get the iPhone app](docs/ios-build.md) | Codemagic, GitHub Actions or Xcode, then sideloading |
| [User guide](docs/using.md) | the window, calibration, tuning, troubleshooting |
| [Raw data](docs/data.md) | Python API, command line, UDP, browser stream, wire format |
| [How it works](docs/how-it-works.md) | architecture, latency, and how the jitter was fixed |

## Related projects

Others have connected phones and hands to computers before. [Gesture](https://github.com/Gesture-App/gesture)
(2022) drove a computer's mouse from an iPhone over Bluetooth, sending updates about ten times a second.
[TrackOSC](https://github.com/JGL/TrackOSC) and [VisionOSC](https://github.com/LingDong-/VisionOSC)
stream Apple Vision results to creative tools over OSC. [Record3D](https://github.com/marek-simonik/record3d)
streams iPhone depth video over USB. HandPilot's angle is latency: a high frame rate, a fast shutter,
inference next to the sensor, and the cable instead of the radio.

## License

[MIT](LICENSE). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

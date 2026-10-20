# Raw data

The mouse is one use of the hand stream. This page is for every other one: your own code, creative
tools, recordings, experiments.

Every frame from the phone carries the 21 landmarks of one hand, with a confidence for each, at up
to 120 frames per second.

## Five ways in

| | Use it when | Mouse involved? |
|---|---|---|
| [Python API](#python) | you're writing Python | no |
| [Command line](#command-line) | you want JSON or CSV on stdout or in a file, for any language | no |
| [UDP](#udp) | the HandPilot window is open and another program wants the data | only if you switch it on |
| [Browser stream](#browser) | a web page (p5.js, three.js…) wants the data | only if you switch it on |
| [Wire format](#the-wire-format) | you want to talk to the phone yourself, in any language | no |

The phone serves **one** program at a time. The Python API and the command line connect to the phone
directly, so close the HandPilot window first. While the window is open, use UDP or the browser
stream: the window receives the data and passes it on.

The Python API and the command line work on Windows, macOS and Linux. Over USB they need Apple's
usbmuxd: on Windows the Apple Devices app or iTunes, built into macOS, and the `usbmuxd` package on
Linux. Or pass a Wi-Fi address.

## A frame

```json
{
  "seq": 812,
  "t": 51234.123456,
  "hand": true,
  "handedness": "right",
  "width": 1920,
  "height": 1080,
  "landmarks": [[0.51234, 0.63412, 0.93], [0.48871, 0.60902, 0.91], "… 21 in all"]
}
```

| Field | Meaning |
|---|---|
| `seq` | frame counter from the phone; gaps mean dropped frames |
| `t` | when the camera captured the frame, in seconds on the **phone's** clock. Use differences between frames, not wall-clock time |
| `hand` | whether a hand was found. When false, `landmarks` is empty |
| `handedness` | `"left"`, `"right"` or `"unknown"`, as Apple Vision reports it |
| `width`, `height` | size of the image the coordinates refer to, after rotation |
| `landmarks` | 21 × `[x, y, confidence]` |

**Coordinates** are 0..1 across the image, origin at the **top left**, as the camera sees you. They
are not mirrored: your right hand appears on the left, like another person looking at you. To measure
distances in equal units on both axes, multiply x by `width / height` first.

**Confidence** is Vision's 0..1 confidence for that point. Below about 0.3 the point is a guess.

### The 21 landmarks

Same order as MediaPipe, so code written for MediaPipe hands works unchanged.

| # | name | # | name | # | name |
|---|---|---|---|---|---|
| 0 | `wrist` | 7 | `index_dip` | 14 | `ring_pip` |
| 1 | `thumb_cmc` | 8 | `index_tip` | 15 | `ring_dip` |
| 2 | `thumb_mcp` | 9 | `middle_mcp` | 16 | `ring_tip` |
| 3 | `thumb_ip` | 10 | `middle_pip` | 17 | `little_mcp` |
| 4 | `thumb_tip` | 11 | `middle_dip` | 18 | `little_pip` |
| 5 | `index_mcp` | 12 | `middle_tip` | 19 | `little_dip` |
| 6 | `index_pip` | 13 | `ring_mcp` | 20 | `little_tip` |

Fingertips are the noisiest points, and the thumb tip most of all. The knuckles (`*_mcp`) are the
steadiest, which is why the mouse follows them.

## Python

```python
from handpilot import HandStream

with HandStream() as hands:                  # USB. Wi-Fi: HandStream(wifi="192.168.1.20")
    for frame in hands:
        if frame.hand:
            x, y, confidence = frame.landmark("index_tip")
            print(f"{x:.3f} {y:.3f}")
```

Run it from this repository's folder, or install the package first with `pip install .`.

| | |
|---|---|
| `HandStream(wifi="", port=9871, latest_only=True, timeout=None)` | `latest_only` keeps only the newest frame while you're busy, so a slow loop never falls behind. Use `False` to receive every frame (for recording). With a `timeout`, iteration stops once no frame arrives for that many seconds |
| `for frame in stream` | yields frames as they arrive |
| `stream.read(timeout)` | the next frame, or `None` |
| `stream.latest()` | the newest frame without waiting. Handy inside a render loop |
| `stream.connected`, `stream.error` | connection state, and the reason when it's down |

Each frame is a `handpilot.Frame`: `frame.hand`, `frame.landmark(name_or_index)`, the raw lists
`frame.x`, `frame.y`, `frame.conf`, plus `frame.width`, `frame.height`, `frame.aspect`,
`frame.handedness`, `frame.t_capture`, `frame.seq`, and `frame.to_dict()` for the JSON above.

## Command line

```
python run.py --stream                     # JSON lines on stdout, one frame per line
python run.py --stream --csv --out hand.csv --seconds 30
python run.py --stream --wifi 192.168.1.20
```

Every frame is written, hand or not, so timestamps stay continuous. Pipe it anywhere:
`python run.py --stream | node consume.js`.

## UDP

With the HandPilot window open, switch on **Raw data → UDP**. Every frame is then sent as one JSON
datagram (the shape above, about 1.3 KB) to `127.0.0.1:9872`, or wherever you point it. Anything that
reads UDP can use it: Python, Node, Unity, TouchDesigner, Max, Processing. A datagram arrives whole or
not at all, so there's no framing to handle.

```python
import json, socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 9872))
while True:
    frame = json.loads(sock.recv(65536))
    if frame["hand"]:
        print(frame["landmarks"][8])      # index tip
```

[`examples/udp_receiver.py`](../examples/udp_receiver.py) is a complete version.

## Browser

Switch on **Raw data → Browser stream**, and any page can subscribe:

```js
const hands = new EventSource("http://127.0.0.1:9870/stream");
hands.onmessage = (event) => {
  const frame = JSON.parse(event.data);
  if (frame.hand) console.log(frame.landmarks[8]);   // index tip [x, y, confidence]
};
```

[`examples/browser.html`](../examples/browser.html) draws the live skeleton on a canvas. Open it
straight from disk.

The stream is off by default, because while it's on any page you visit could read your hand
movements. It only ever sends landmarks; the rest of the window's controls need a per-session token.

## The wire format

For talking to the phone yourself, from any language. The phone app is a TCP **server** on port
**9871**. Over Wi-Fi, connect to the phone's IP. Over USB, go through Apple's multiplexer:
[`handpilot/usbmux.py`](../handpilot/usbmux.py) does it in about 200 lines, or run `iproxy 9871 9871`
from libimobiledevice and connect to `localhost:9871`.

Everything from the phone is wrapped in an envelope, little-endian:

| bytes | |
|---|---|
| 4 | magic `HPU1` |
| 1 | type: 1 = frame, 2 = status, 3 = pong |
| 2 | payload length, `uint16` |
| n | payload |

A **frame** payload is `<IddBBBBHHf` followed by 21 × `<fff`:

| type | field |
|---|---|
| `uint32` | sequence number |
| `float64` | capture time, phone clock, seconds |
| `float64` | send time, same clock (send − capture = the phone's processing time) |
| `uint8` | hand present (0/1) |
| `uint8` | chirality: 0 unknown, 1 left, 2 right |
| `uint8` | orientation: 0 portrait, 1 landscape left, 2 landscape right, 3 upside down |
| `uint8` | reserved |
| `uint16` × 2 | width, height |
| `float32` | hand model time, ms |
| `float32` × 63 | x, y, confidence for each landmark (zeros when absent) |

A **status** payload is JSON, about four times a second: frame rates, model time, ISO, shutter,
temperature (`thermal` 0–3), the active camera and format, and under `settings` the camera settings
in the same keys you send them.

**Commands to the phone** are newline-terminated JSON objects. Unknown keys are ignored:

```json
{"fps": 120, "shutter": 500, "camera": "front", "crop": true, "zoom": 1.5}
{"iso_auto": true, "iso_ceiling": 4000, "focus_auto": false, "lens": 0.6}
{"orientation": 2, "preview": false}
{"ping": 7}
```

`ping` is echoed back as a pong; HandPilot uses it to measure the cable's round trip.
[`handpilot/protocol.py`](../handpilot/protocol.py) is the reference decoder, and
[`ios/HandPilot/HandPilot/Wire.swift`](../ios/HandPilot/HandPilot/Wire.swift) the encoder.

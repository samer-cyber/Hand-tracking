# How to use this

Practical reference: what to run, what the numbers mean, what it can't do.

---

## Start here — phone only, no camera calibration

```bash
python run_hand.py
```

The phone alone is a complete tracker. It measures 21 landmarks, a depth at
each, and reports its own intrinsics, so metric 3D needs no camera calibration
at all: a pixel fixes the direction, the depth fixes the distance.

Start HandBridge on the phone first (address from `python tools/whats_my_ip.py`,
port 9870).

### Calibrating the screen

The one calibration that remains is finding your *screen* in 3D. Point your
index finger at each corner and press SPACE — all four corners, twice, from two
hand positions **at least 30 cm apart**.

The second pass is not busywork. One ray per corner fixes only a direction; the
distance to the screen stays unknown. Two rays from different places intersect
at a point, which is what actually locates it. Measured: passes 30 cm apart give
25 mm corner accuracy, 3 cm apart give 150 mm, 1 cm apart give 423 mm.

When it finishes it prints the screen size it recovered. **Your screen is 15.5
inches — if it reports something close, the calibration is good.** More than a
couple of inches off means redo it with `c`, aiming more carefully.

### Pointing modes — press `m` to cycle

Measured on a 15.5-inch screen at 40 cm with 2 mm of tracking jitter
(`python tools/test_pointing.py`):

| mode | cursor jitter | hand travel to cross screen | feel |
|---|---|---|---|
| **absolute** | **7 px** | 47 cm | steadiest; hand must stay in a region |
| relative | 8 px | 72 cm | trackpad with gain; works from anywhere |
| eye ray | 28 px | 36 cm | point naturally; less precise |
| finger ray | 100 px | 39 cm | most direct, least steady |

Try them in that order. **Absolute is four times steadier than the best ray**,
because a ray amplifies hand jitter by roughly (distance to screen) / (lever
arm), while absolute mapping does not amplify at all.

This contradicts published studies that find ray-casting better — those measure
VR, where targets are metres away and physically unreachable so absolute mapping
is impossible. At a desk the screen is within arm's reach and the geometry
reverses. Whether it *feels* better is yours to judge; that is why all four are
switchable at runtime.

### Controls

| key | does | works unfocused |
|---|---|---|
| `m` | cycle pointing mode | preview window |
| `SPACE` | record a calibration sample | preview window |
| `c` | redo screen calibration | preview window |
| `p` | prediction on/off | preview window |
| **`F9`** | cursor control on/off | **yes** |
| **`F10`** | clicking on/off | **yes** |
| **`ESC`** | emergency stop | **yes** |
| `q` | quit | preview window |

Cursor control and clicking start **off**. Turn them on deliberately: once a
tracker owns your pointer, the usual way to stop it is clicking, which is the
thing you have lost control of. F9/F10/ESC read the keyboard directly so they
work whatever is in front.

Pinch thumb to index to click; hold to drag.

---

## Adding the laptop camera (optional)

## Where to put the phone

Computed by searching placements against coverage and triangulation angle
(`python tools/plan_rig_geometry.py` re-runs it). Measured **from the laptop
camera**, which sits centre-top of your screen bezel:

| | distance |
|---|---|
| to the RIGHT | **46 cm** |
| BELOW it | **14 cm** |
| FORWARD, toward you | **18 cm** |

Aim the phone **64° to the left** (rotated to face the space in front of your
screen), tilted roughly level. **Portrait**, front camera toward your hands.

Left-handed or prefer it on the left? Mirror the horizontal number; nothing else
changes.

**This is forgiving.** Being 5 cm off in any direction changes predicted error
by under 0.1 mm, and aiming up to 20° off-target barely matters. Get it roughly
right and stop worrying — no 3D-printed mount needed yet.

What that placement buys: **63% of the working volume seen by both cameras, 56°
parallax, ~0.9 mm predicted error.** For comparison, your measured overlap so
far has been 1–23%, which is why most frames fell back to a single camera.

**The laptop camera is the limit, not the phone.** Its ~65° field of view only
covers 64% of the volume where hands naturally sit, so 63% is close to the
ceiling. If you want more, keep your hands nearer the screen centre.

---

## Running it

### Everything at once

```bash
python tools/doctor.py --laptop 0 --phone 2
```

Checks the whole chain and stops at the first real problem. Run this first when
anything misbehaves. The line that matters most is **"both cameras saw the hand
together"** — below ~40% and stereo can't do its job regardless of the code.

### Tracking

```bash
python run_fusion.py --laptop 0 --phone 2
```

| key | does |
|---|---|
| `q` | quit |
| `x` | realign the phone to the laptop (fixes a drifted rig) |
| `r` | cycle raw / processed / both skeletons |
| `p` | toggle forward prediction |
| `f` | freeze the hand model (makes bone-residual a drift detector) |
| `h` | hide the HUD |

### Mouse control

```bash
python run_mouse.py --laptop 0 --phone 2
```

First run asks you to point at each screen corner and press SPACE. **Hold your
hand where it's comfortable — don't stretch.** That region becomes the whole
screen.

| key | does | works unfocused |
|---|---|---|
| `F9` | cursor control on/off | yes |
| `F10` | clicking on/off | yes |
| `ESC` | emergency stop | yes |
| `q` | quit | preview window only |

Both start **off** on purpose. Once a tracker owns your pointer, the usual way
to stop it is clicking — which is the thing you've lost control of. F9/F10/ESC
read the keyboard directly so they work regardless of what's in front.

Pinch thumb to index to click; hold to drag.

---

## Reading the HUD

| field | healthy | meaning |
|---|---|---|
| `mode` | `laptop+phone` | which sensors fed this frame |
| `latency` | 30–60 ms | frame arrival → result ready |
| `reprojection` | under 3 px | how well the two views agree |
| `bone residual` | under 2 mm | how metrically right the rig is |
| `uncertainty` | 1–3 mm | the filter's own confidence |
| `both views` | above 40% | how often stereo is possible |
| `3D jitter` | under 2 mm | noise while holding still |

**`reprojection` and `bone residual` measure different things.** Reprojection
can be excellent while the rig is metrically wrong, because two views alone
can't determine scale — a rig twice as big viewing a hand twice as far away
projects identically. Bone residual is what catches that, since your bones have
a known length.

---

## Fixing a bad rig

If `bone residual` is high or the hand sits in the wrong place:

1. In `run_fusion.py`, press **`x`**, then move your hand slowly where both
   cameras see it for ~90 frames.
2. Check the printed **baseline** (should be plausible for where the phone
   physically is) and **bone residual** (want under 2 mm).

If that doesn't help, redo the calibration from scratch:

```bash
python tools/calibrate_rig_hand.py --cam-a 0 --cam-b 2
```

Watch **coverage** (get it above 50% by moving your hand into all corners of
both views) and **scale IQR** (want under 0.05; 0.15+ means the result is
unreliable).

---

## Limitations, honestly

**The laptop camera caps at 30 fps** and its exposure can't be controlled — gain
isn't settable, so forcing a short shutter just yields a black frame. More light
on your hands is the only lever, and it genuinely helps: 3D error scales
linearly with landmark pixel noise.

**Your laptop has no IR camera.** Confirmed at the hardware level: the LOQ
15IRX9 exposes one Realtek "Integrated Camera" and no Windows Hello IR device.
The 1080p grayscale stream at index 1 returns pure black and always will.

**The iPhone's IR is available only as depth, not as an image.** TrueDepth
projects an infrared dot pattern, but the raw IR frame is reserved for Face ID
inside the Secure Enclave. Third-party apps get the depth map derived from it —
which is the useful part anyway.

**TrueDepth range is about 15 cm to 1 m.** Past that it returns nothing, so keep
hands within ~60 cm of the phone.

**Occlusion still breaks things.** Two views help; a fist seen edge-on by both
does not resolve.

**Reference points:** Quest 3 hand tracking measures ~70 ms, Vision Pro ~128 ms.
Vision Pro feels good despite being the slower of the two, which says most of
"feel" comes from prediction and interaction design rather than raw latency.

---

## Measured on this rig

The phone turned out to be far better than assumed, and two design decisions
were made on guesses that measurement overturned.

| | measured | I had assumed |
|---|---|---|
| phone frame rate | **65 fps** | 30 fps |
| phone transit (capture -> PC) | **12.8 ms** median, 20.8 p95 | ~120 ms feared |
| packet loss | **0** in 1910 packets | unknown |
| depth when present | **21/21** landmarks | 12+ hoped |
| laptop frame rate | 30 fps (hard cap) | 30 fps |

Two consequences:

**The phone is the primary camera now, not the helper.** It runs at double the
laptop's rate with 13 ms of transit. The laptop's 30 fps cap is the slower half
of the rig, which inverts the original plan.

**Published warnings about Apple's Vision hand pose being slow (~120 ms) did not
hold here.** On-device detection plus Wi-Fi round trip totals ~13 ms, so the
on-phone architecture is vindicated. There is no need to fall back to streaming
frames.

**Depth arrives on roughly half the frames, and that is correct.** Video runs at
65 Hz while the TrueDepth stream runs near 30 Hz, so alternate frames have no
depth partner. Video is deliberately *not* held back to the depth rate: doubling
the landmark rate halves sampling latency, which is the weakest axis of the
system, while depth is an optional measurement to the fusion filter and loses
very little by arriving half as often.

---

## The iPhone app (HandBridge)

Under `ios/`. It runs hand detection **on the phone** and sends ~384 bytes per
frame instead of video — no encode, no decode, no video buffering. It also
reports the camera's **measured** intrinsics, which removes the field-of-view
guess that has corrupted every calibration so far.

Build and install instructions: **[docs/IPHONE_APP.md](docs/IPHONE_APP.md)**.

Once installed:

```bash
python tools/whats_my_ip.py      # the address to type into the app
```

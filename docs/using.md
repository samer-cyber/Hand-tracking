# User guide

## Start it

```
python run.py
```

This opens the HandPilot window, as an Edge or Chrome app window, or a browser tab if neither is
installed. There's nothing to install beyond Python 3.10+. Closing the window quits HandPilot, and
running `python run.py` again while it's open just brings the window back.

The window connects to the phone on its own. Over USB, Windows needs Apple's driver: install the
**Apple Devices** app from the Microsoft Store (or iTunes) and open it once with the phone plugged in.

## Set the phone up

**Back camera (default).** Stand the phone in landscape, below or beside your monitor, with the rear
camera pointing at you, about a forearm's length away (40–70 cm). The back camera has the bigger sensor,
which matters at fast shutter speeds.

**Front camera.** Choose *Front* on the phone or in the window's Camera card. The phone then faces you
like a webcam, and its screen can show the live skeleton. Its sensor is smaller, so it needs more light
for the same sharpness.

Landscape matters more than it sounds: a widescreen monitor needs the camera's wide side
left-to-right. If the skeleton looks rotated, set **Phone orientation** explicitly. A phone lying in a
stand often reports "face up", which gives *Automatic* nothing to go on.

**Light the room.** At 1/500 s the camera gathers a fraction of the light it normally would, and the
ISO it boosts to compensate turns into noise, and noise into jitter. If the **Light** tile says *too
dark for this shutter*, add light or pick 1/250 s. A well-lit room makes a bigger difference than any
setting.

## The window

**Top bar.** Connection state, and the mode: **Off** (tracking only), **Mouse**, or **Gestures** if
you've enabled it. **F9** releases control from anywhere and **F10** toggles Mouse. Both work while the
window is in the background.

**Live view.** The hand as the phone sees it, mirrored so it matches your movements. The dashed box
is the **screen area**: the part of the camera's view that maps onto your monitor. The blue ring is the
point the cursor follows. The skeleton turns amber when the model is unsure about the hand, which is the
cue for more light or a better angle.

**Tiles along the bottom:**

| Tile | Meaning |
|---|---|
| Latency | estimated time from camera to cursor, measured live |
| Tracking | frames per second the phone tracks, out of those it captures |
| Hand model | how long Apple's hand model takes per frame, on the phone |
| Stillness | how much the cursor wobbles while your hand is still (Mouse mode). Raw wobble is shown below it |
| Light | camera ISO; warns when the room is too dark for the shutter speed |
| iPhone | temperature: *Cool*, *Warm*, *Hot*, *Too hot* |

## Calibrate

**Calibrate screen area**, then sweep your hand over the area you want to use for six seconds. The green
box grows as you go and becomes your screen. Choose a range you can cover without moving your shoulder:
a big area is precise but tiring, and a small one is fast but twitchy. Calibrate again if you move the
phone or change *Cursor follows your*.

## Tune it, in this order

1. **Steadiness.** *Balanced* suits most people. Hold your hand still in Mouse mode and watch
   **Stillness**. If the cursor still creeps, choose *Steady*. If it feels like it trails your hand,
   choose *Responsive*.
2. **Click sensitivity.** The bar shows the live gap between thumb and index. The green mark is where a
   click starts. Pinch a few times and set the slider so a deliberate pinch crosses the mark and a relaxed
   hand stays well clear.
3. **Cursor follows your knuckles or fingertips.** Knuckles are much steadier and don't move when you
   pinch. Fingertips put the cursor between your fingers, which some people find more direct.
4. **Frame rate.** 120 fps is the sweet spot. 240 halves the light per frame and the hand model can't
   keep up. 60 runs cooler.
5. **Hand crop** (on by default) runs the hand model on a crop around your hand rather than the whole
   picture, for sharper landmarks. Compare it on and off with the Stillness tile.

Everything else lives under **Advanced**: connection, which monitor, mirroring, raw filter values,
click timing, and experimental gestures. **Reset mouse settings** restores the defaults and keeps
your calibrated area.

## Clicking and dragging

- **Click.** Pinch thumb and index, then release. The cursor freezes while you pinch so a click can't
  slide into a drag.
- **Drag.** Pinch, then move. Beyond the *click freeze radius* the cursor unfreezes and drags.
- **Right click.** Pinch thumb and middle finger (toggle in the Click card).
- **Scroll.** Index and middle up, the others curled, then move up or down (toggle in the Click card).
- If tracking drops out mid-drag, the button is held briefly in case the hand comes back, then released.

## Where settings live

Desktop settings are saved to `config.json` next to `run.py` (or `%APPDATA%\HandPilot\config.json`
when installed as a package). The path is shown at the bottom of *Advanced*. Camera settings are stored
on the phone, because the phone applies them.

## Troubleshooting

| What you see | What to do |
|---|---|
| *Waiting for your iPhone* | Click **Diagnose connection**: it checks each link and names the fix |
| "Apple's USB service is not reachable" | Install the Apple Devices app from the Microsoft Store, plug the phone in, open the app once |
| "No iPhone found on USB" | Use a data cable (some only charge), unlock the phone, tap **Trust** |
| "The iPhone didn't answer" | Open HandPilot on the phone and tap **Start**; check the port matches (9871) |
| Connected, but no hand | Hold your hand up in view of the camera; turn up the light |
| Skeleton rotated or upside down | Set **Phone orientation** explicitly (Camera card) |
| Cursor moves the wrong way | Toggle **Mirror left and right** (Advanced → Mapping) |
| Cursor creeps while you hold still | *Steady* preset; more light; turn on Hand crop |
| Cursor trails behind fast moves | *Responsive* preset |
| Clicks fire by themselves | Lower the click sensitivity |
| Clicks turn into small drags | Raise the click freeze radius (Advanced → Clicks and tracking) |
| Can't reach the screen corners | Calibrate again with a bigger sweep |
| Tracking well below capture fps | Drop to 120 or 60 fps |
| iPhone says *Hot* | Drop to 60 fps; switch the phone screen to *Skeleton* (cheaper than *Camera*) |
| The mouse has run away | **F9** |

The same checks from the command line: `python run.py --doctor`.

# Setting up HandPilot

Two halves. The phone app has to be built and sideloaded once (about 30 minutes,
and you have done this before for HandBridge). The desktop app runs from source
with nothing to install.

HandPilot uses its own bundle id, so it installs **alongside** HandBridge rather
than replacing it. Nothing you already have stops working.

---

## Part 1 — Build the phone app

You have no Mac, so it is built on GitHub's macOS runners, exactly as before.

### 1. Push this project to GitHub

```bash
cd "D:\Samer\Projects\Hand tracking 2"
git init
git add .
git commit -m "handpilot"
```

Make a repo on github.com, then:

```bash
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

A **public** repo gets unlimited free macOS minutes. A private one works too and
just eats your monthly quota at 10× — a build is a few minutes, so it lasts a
while either way.

### 2. Run the build

On GitHub: **Actions** → **Build HandPilot (unsigned IPA)** → **Run workflow**.

Three to six minutes. Download the **HandPilot-unsigned-ipa** artifact from the
run page and unzip it to get `HandPilot-unsigned.ipa`.

> If it fails, open the failed step and send me the error. This is Swift written
> without a Mac to compile against, so a first-attempt failure is normal and
> usually a one-line fix — that is what the CI is for.

### 3. Sideload it

Same as HandBridge: Sideloadly, drag in the `.ipa`, enter your Apple ID, Start.

Developer Mode and the Apple-ID trust are already on your phone from last time,
so there is nothing extra to enable. Same 7-day signature expiry as before.

---

## Part 2 — The Windows side

Nothing to install. Python 3.10+ with tkinter, which the python.org installer
gives you by default.

```bash
cd "D:\Samer\Projects\Hand tracking 2"
python run.py
```

### One thing you do need: Apple's USB service running

You already have the **Apple Devices** app from the Microsoft Store — good, it
contains the USB multiplexer this depends on. But it is **not running right now**,
and it only starts when the app is opened or a device is plugged in.

So: **plug the iPhone in and open the Apple Devices app once.** After that it
stays up in the background and you can close the window.

To check at any time:

```bash
python run.py --doctor
```

It walks the whole path — Python, the Apple service, the cable, the app, the data
rate — and tells you the specific fix for whatever is missing.

If you would rather skip all of this, switch the **Phone** bar in HandPilot to
**Wi-Fi** and type the phone's IP. It works, it just adds a few milliseconds of
radio latency and some jitter.

---

## Part 3 — Physically setting it up

**Put the phone in landscape**, standing just below or beside your monitor,
rear camera pointing at you, roughly 40–70 cm away.

Landscape matters more than it sounds. In portrait the camera's field of view is
tall and narrow, so mapping it onto a widescreen monitor means a small horizontal
range doing all the work. Landscape gives you the wide sweep the screen actually
needs.

**Light the room.** This is the one real constraint. A 1/500 s shutter is about
three stops darker than a phone's default, and the ISO servo compensates by
amplifying — which amplifies noise, and noisy pixels make noisy landmarks. On the
Camera tab, if **iso** sits pinned at its ceiling, you are asking for more than
the room can give: turn a light on, or slow the shutter to 1/300 or 1/250. The
tracking quality difference between a well-lit room and a dim one is larger than
any setting in this app.

---

## Part 4 — First run

1. Open **HandPilot** on the phone. Check the port says `9871`. Press **Start**.
   Aim it at yourself using the preview, then turn it around and set it down.
2. On the PC, run `python run.py`. The **Phone** bar should go green within a
   second or two.
3. Hold a hand up. The panel on the right should draw a skeleton.

**If the skeleton looks rotated**, set **Mount** on the Camera tab explicitly to
Landscape left or Landscape right instead of Auto. A phone lying in a stand often
reports "face up" rather than a real orientation, so Auto has nothing to go on.

**If it moves the wrong way**, that is the **Flip X** / **Flip Y** checkboxes on
the Pointer tab. Flip X is on by default because the camera faces you and
therefore sees you the way another person would.

### Calibrate the pointer

Pointer tab → **Calibrate**, then move your hand around the area you want to map
to your screen for six seconds. Sweep the whole comfortable range — the box drawn
on the preview is what becomes your monitor.

Pick a range you can reach without moving your shoulder. A big region is precise
but tiring; a small one is fast but twitchy.

### Try it

Pick **Pointer (hand as mouse)** at the top.

- The cursor follows the midpoint between your thumb and index fingertip.
- **Pinch thumb and index** to click. The cursor freezes in place while pinched so
  a click cannot slide into a drag — but move more than the freeze radius and it
  unfreezes and becomes a real drag.
- **Pinch thumb and middle** for a right click.
- **F9** releases control instantly, from anywhere.

Then try **Gesture control**. Five gestures are bound out of the box:

| gesture | does |
|---|---|
| Thumbs up | volume up (repeats while held) |
| Thumbs down | volume down (repeats while held) |
| Fist | play / pause |
| Swipe right | next track |
| Swipe left | previous track |

Ten more are ready and switched off — OK sign, peace, pointing, pinch, rock
horns, shaka, three fingers, open palm, swipe up, swipe down. Tick one on and
pick from about sixty actions: volume, media, clicks, scrolling, slides, window
management, browser, editing, mode switching, any custom hotkey, or running a
command.

---

## Tuning, in the order worth doing it

**1. Pinch thresholds.** Watch the `pinch` number under the live view. Pinch and
unpinch a few times and note the two values. Set *Pinch closes at* just above your
pinched number and *Pinch opens at* comfortably below your open one. Getting this
right fixes more perceived flakiness than anything else.

**2. Click freeze radius.** Default 16 px. Raise it if clicks turn into small
accidental drags; lower it if deliberate drags feel like they need too much
movement to start.

**3. Smoothing vs Responsiveness.** *Smoothing* controls steadiness while your
hand is still; *Responsiveness* controls how much the filter opens up as you move.
If the cursor is jittery when you hold still, lower Smoothing. If it lags behind
fast movement, raise Responsiveness. They are separate on purpose — a single
control cannot fix both.

**4. Lead ahead.** Start at 0. Add 10–20 ms and it starts feeling like the cursor
arrives with your hand. Too much and it overshoots when you change direction.

**5. Zoom.** If your hand is small in frame, 1.5× crops in. Vision downsamples the
image before it looks at it, so a hand filling more of the frame is genuinely more
landmark detail, not just a bigger picture.

---

## When something is wrong

| what you see | what it is |
|---|---|
| "Apple's device service is not running" | open the Apple Devices app once with the phone plugged in |
| "connection refused" | HandPilot is not running on the phone, or its port is not 9871 |
| connected, no frames | press Start in the phone app |
| skeleton rotated 90° | set Mount explicitly on the Camera tab |
| cursor moves the wrong way | Flip X / Flip Y on the Pointer tab |
| skeleton is amber, not green | landmarks arrived but were too uncertain to act on — more light, or lower the confidence floor |
| clicks that should be clicks turn into drags | raise the click freeze radius |
| cursor jitters when still | lower Smoothing |
| cursor lags behind fast movement | raise Responsiveness, then add Lead ahead |
| gestures fire when you did not mean them | raise that gesture's hold time |
| tracked fps well below capture fps | Vision cannot keep up — drop to 60 fps |
| thermal says *serious* | drop to 60 fps, turn off the phone preview |
| iso pinned at its ceiling | the room is too dark for that shutter |
| the mouse has run away | **F9** |

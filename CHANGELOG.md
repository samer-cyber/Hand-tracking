# Changelog

## 1.0.0 — 2026-10-20

First public release.

**Smoother pointer**
- The cursor follows the knuckles instead of the thumb and index tips: much steadier, and it no longer
  moves when you pinch to click. Fingertips remain an option.
- One **Steadiness** control (*Responsive*, *Balanced*, *Steady*) replaces the two filter sliders.
  The old "Smoothing" slider worked backwards.
- A dead zone holds the cursor perfectly still while your hand is still.
- The filter runs on the phone's capture timestamps rather than bursty arrival times.
- Lead-ahead is off by default; it overshot every time the hand stopped.
- Maps onto the main monitor by default, not the whole multi-monitor desktop.
- A **Stillness** readout measures cursor wobble, to compare settings objectively.

**iPhone app**
- **Front camera** option, with Center Stage disabled and orientation handled per model.
- **Hand crop**: the hand model runs on a crop around the hand for sharper landmarks.
- Redesigned dark interface with a live skeleton view.
- Reports its camera settings, so the computer always shows what the phone is actually doing.
- Builds on **Codemagic** (`codemagic.yaml`) as well as GitHub Actions; tagged releases get the `.ipa`.

**Desktop app**
- New window, served locally and opened as an app window: connection guidance and **Diagnose**, a
  live view with the screen area, a calibration flow, live metrics, and phone camera controls.
- **Raw data** without the mouse: `HandStream` Python API, `python run.py --stream` (JSON or CSV),
  UDP output, and an opt-in browser stream.
- Gestures are now experimental and off by default.
- Settings from earlier versions migrate automatically. Filter values, lead-ahead and click
  thresholds start from the new defaults, and the window asks for one calibration.

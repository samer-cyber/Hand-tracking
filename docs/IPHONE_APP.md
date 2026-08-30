# Getting HandBridge onto your iPhone

You have no Mac, so the app is built on GitHub's macOS runners (free) and
installed with a sideloader that signs it using your own Apple ID.

Budget about 30–40 minutes the first time. After that, rebuilding is one click.

---

## 1. Put this project on GitHub

```bash
git init
git add .
git commit -m "hand tracking"
```

Create a repo on github.com, then:

```bash
git remote add origin https://github.com/YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

A **public** repo gets unlimited free macOS runner minutes. A private one uses
your monthly quota, and macOS minutes count at 10× — a build is a few minutes,
so a private repo works too, just not indefinitely.

## 2. Build it

On GitHub: **Actions** tab → **Build HandBridge (unsigned IPA)** → **Run
workflow**.

Takes 3–6 minutes. When it finishes, download the **HandBridge-unsigned-ipa**
artifact from the run page and unzip it to get `HandBridge-unsigned.ipa`.

**If the build fails**, open the failed step and copy the error to me. I wrote
this Swift without a Mac to compile against, so a first-attempt failure is
likely and easy to fix — that's exactly what this CI is for.

## 3. Install it on the phone

Use **Sideloadly** (simplest) — <https://sideloadly.io>, Windows version.

1. Install Sideloadly and iTunes if you don't have it (Sideloadly needs Apple's
   USB drivers, same as before).
2. Plug in the iPhone, unlock it, tap **Trust**.
3. Open Sideloadly, drag in `HandBridge-unsigned.ipa`.
4. Enter your Apple ID. It signs the app locally with a free developer
   certificate — Apple never sees the app, and your Apple ID stays on your
   machine.
5. Click **Start**.

Then on the phone: **Settings → General → VPN & Device Management** → tap your
Apple ID → **Trust**. iOS will not launch a sideloaded app until you do this.

### The 7-day thing

A free Apple ID signs apps for **7 days**. After that HandBridge refuses to
launch and you re-run Sideloadly (the ipa stays valid; only the signature
expires). **AltStore** can refresh it automatically over Wi-Fi if the weekly
step gets annoying.

A paid Apple Developer account ($99/yr) extends this to a year. Not worth it
unless this becomes something you use daily.

## 4. Connect it

On the PC:

```bash
python tools/whats_my_ip.py
```

Type that address and port **9870** into HandBridge, press **Start**.

The app shows what matters:

| row | want |
|---|---|
| link | `connected` |
| camera | `streaming` |
| rate | 25–30 fps |
| hand | `tracking` when your hand is in view |
| depth | 12+/21 landmarks |

**If `link` says connected but the PC receives nothing**, it's almost always the
Windows firewall. Allow Python inbound UDP on Private networks. Both devices
must also be on the same Wi-Fi — phone on cellular won't reach your PC.

**If `depth` stays low**, you're too far away. TrueDepth works from roughly
15 cm to 1 m and returns nothing beyond that.

---

## Why an app instead of a webcam stream

Three concrete reasons, in order of how much they matter:

**Measured intrinsics.** The camera reports its own focal length and optical
centre every frame. Every calibration so far has had to assume a field of view,
and a wrong assumption corrupts the essential-matrix solve that the rig geometry
depends on — producing a rig that looks self-consistent and is metrically wrong.
That was the root of the wandering baselines (61 cm, then 25 cm, then 9 cm).

**No video in the path.** 384 bytes per frame instead of a compressed image
means no encoder, no decoder, no video buffering. Phone-as-webcam latency was
the largest unquantified term in the budget.

**Real depth.** TrueDepth distance in metres at each landmark, which is what
keeps tracking alive when the laptop camera loses your hand.

## What it does not do

It sends no image, so the PC cannot run its own detector on the phone's view —
you get Apple's Vision hand landmarks, not MediaPipe's. Published estimates put
Vision's hand pose latency near 120 ms, which would be poor. **This needs
measuring before trusting it**, and the app timestamps every frame so we can.
If Vision turns out to be too slow or too imprecise, the fallback is streaming
compressed frames instead, which the same app can be extended to do.

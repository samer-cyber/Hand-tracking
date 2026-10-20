# Get the iPhone app

HandPilot isn't on the App Store. You build it from this repository and install it with your own
Apple ID. You don't need a Mac or a paid developer account.

| Way | You need | Time |
|---|---|---|
| [**Codemagic**](#a-codemagic) | a free Codemagic account | ~10 minutes the first time |
| [**GitHub Actions**](#b-github-actions) | a GitHub account | ~10 minutes |
| [**Xcode**](#c-xcode-on-a-mac) | a Mac | ~5 minutes |

If this repository's *Releases* page has a `HandPilot-unsigned.ipa`, you can download that and skip
straight to [Install it](#install-it).

All three build from the same source. The Xcode project is generated from
[`ios/HandPilot/project.yml`](../ios/HandPilot/project.yml) by [XcodeGen](https://github.com/yonaskolb/XcodeGen),
so there is no project file to drift out of date.

---

## A. Codemagic

[Codemagic](https://codemagic.io) is a build service with Mac machines. Its free plan includes
500 build minutes a month, and one HandPilot build takes about five.

1. Put this repository on your own GitHub, GitLab or Bitbucket account (a fork is fine).
2. Sign in to [codemagic.io](https://codemagic.io) with that account and choose **Add application**.
   Pick the repository. Codemagic finds [`codemagic.yaml`](../codemagic.yaml) in it on its own.
3. Click **Start new build**, choose the workflow **HandPilot iPhone app (unsigned IPA)**, and start it.
4. When it turns green, open **Artifacts** and download **HandPilot-unsigned.ipa**.

Then [install it](#install-it).

## B. GitHub Actions

1. **Fork** this repository on GitHub. Public repositories get free macOS build minutes.
2. In your fork, open the **Actions** tab. If GitHub asks, enable workflows.
3. Choose **Build iPhone app (unsigned IPA)** → **Run workflow**.
4. Open the finished run, download the **HandPilot-unsigned-ipa** artifact, and unzip it to get
   `HandPilot-unsigned.ipa`.

Then [install it](#install-it). Pushing a version tag (for example `v1.0.0`) also attaches the
`.ipa` to that GitHub release, so other people can skip building.

## C. Xcode on a Mac

```bash
brew install xcodegen
cd ios/HandPilot
xcodegen generate
open HandPilot.xcodeproj
```

In Xcode, select the **HandPilot** target → **Signing & Capabilities**. Choose your team (a free
personal Apple ID works), and if Xcode complains, change the bundle identifier to something unique such
as `com.yourname.handpilot`. Plug the iPhone in, pick it as the destination, and press **Run**.
Xcode installs it for you, so skip the next section.

---

## Install it

An unsigned `.ipa` needs signing with your Apple ID as it's installed. **Sideloadly** does both, on
Windows or macOS. [AltStore](https://altstore.io) works too.

1. **Windows only:** install the free **Apple Devices** app from the Microsoft Store, or iTunes, so
   the PC can see the iPhone.
2. Plug the iPhone in, unlock it, and tap **Trust** if it asks.
3. Open [Sideloadly](https://sideloadly.io), drag `HandPilot-unsigned.ipa` onto it, enter your Apple
   ID, and click **Start**.
4. On the iPhone, the first time only:
   - **Settings → Privacy & Security → Developer Mode → On**, then restart when asked.
   - **Settings → General → VPN & Device Management**: tap your Apple ID and **Trust** it.

With a free Apple ID the app stops opening after **7 days**. Run Sideloadly again to refresh it;
your settings are kept. Free accounts can have three sideloaded apps at a time.

## First launch

Open **HandPilot** and tap **Start**.

- Allow **camera** access. Nothing is recorded or uploaded; the camera feeds the hand model and
  nothing else.
- If iOS asks about **finding devices on your local network**, allow it. It's needed for the Wi-Fi
  connection; the USB cable works either way.

Then follow the [user guide](using.md) on your computer.

## If the build fails

The build log names the file and line. Open an issue with that excerpt. The Swift is compiled by
Xcode on the build machine, and newer Xcode releases occasionally rename an API.

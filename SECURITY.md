# Security and privacy

## What leaves the phone

Only hand landmarks: 21 points per frame with confidences, plus status numbers such as frame rate,
ISO and temperature. Camera frames are never sent, stored or uploaded. Nothing goes to the internet.

## Things to know

- **Wi-Fi mode** listens on port 9871 on your local network with no password. Anyone on the same
  network could read your hand landmarks or change the phone's camera settings. Use the USB cable, or
  Wi-Fi only on networks you trust. Tapping **Stop** in the iPhone app closes the port.
- **The desktop window** is a local web server on `127.0.0.1` only. Every API call needs a token
  generated for that run, and the Host header is checked, so other web pages you have open can't
  drive your mouse.
- **UDP output** and the **browser stream** are off by default. While on, any program on your computer
  can read your hand movements, and with the browser stream, any web page can too.
- **Gestures** (experimental, off by default) can be bound to "Run a command…", which runs whatever you
  type there when you make that gesture. Only enter commands you trust.
- **F9** releases mouse control from anywhere.

## Reporting a vulnerability

Please use GitHub's **Report a vulnerability** (Security tab → Advisories) rather than a public issue.

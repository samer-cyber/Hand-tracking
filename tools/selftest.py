"""Exercise the desktop half without a phone.

Synthesises hands, poses and a fake phone on a socket, then checks that the
feature extraction, the pose classifier, the wire format and the link all agree.
Worth having because the alternative way to find out that a threshold is wrong is
to stand in front of a camera waving, which tells you *that* something is broken
and never *which* thing.

Nothing here touches the mouse or the keyboard.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handpilot import gestures, handmodel, protocol           # noqa: E402
from handpilot.filters import Hysteresis, OneEuro2D           # noqa: E402
from handpilot.link import Link                               # noqa: E402
from handpilot.protocol import Frame                          # noqa: E402

WIDTH, HEIGHT = 1920, 1080
ASPECT = WIDTH / HEIGHT

PASS, FAIL = 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [ ok ] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# --------------------------------------------------------------------- hands

def make_hand(ext=(0.9, 0.9, 0.9, 0.9, 0.9), thumb_dir=(-0.7, -0.7),
              pinch=None, centre=(0.5, 0.5), scale=0.18, conf=0.95) -> Frame:
    """A synthetic right hand, upright, in normalised image coordinates.

    Built in units of image height and converted back at the end, which is the
    same correction ``handmodel`` applies -- so a hand built here with a given
    finger extension should read back with that extension.
    """
    cx, cy = centre[0] * ASPECT, centre[1]
    s = scale

    px = [0.0] * 21
    py = [0.0] * 21
    px[handmodel.WRIST], py[handmodel.WRIST] = cx, cy + 0.55 * s

    knuckles = {handmodel.INDEX_MCP: -0.30, handmodel.MIDDLE_MCP: -0.10,
                handmodel.RING_MCP: 0.10, handmodel.LITTLE_MCP: 0.30}
    for joint, offset in knuckles.items():
        px[joint], py[joint] = cx + offset * s, cy
    px[handmodel.THUMB_CMC], py[handmodel.THUMB_CMC] = cx - 0.42 * s, cy + 0.42 * s
    px[handmodel.THUMB_MCP], py[handmodel.THUMB_MCP] = cx - 0.52 * s, cy + 0.22 * s

    palm = max(math.hypot(px[handmodel.WRIST] - px[handmodel.MIDDLE_MCP],
                          py[handmodel.WRIST] - py[handmodel.MIDDLE_MCP]),
               math.hypot(px[handmodel.INDEX_MCP] - px[handmodel.LITTLE_MCP],
                          py[handmodel.INDEX_MCP] - py[handmodel.LITTLE_MCP]))

    chains = (
        ("index", handmodel.INDEX_MCP, handmodel.INDEX_PIP,
         handmodel.INDEX_DIP, handmodel.INDEX_TIP, ext[1]),
        ("middle", handmodel.MIDDLE_MCP, handmodel.MIDDLE_PIP,
         handmodel.MIDDLE_DIP, handmodel.MIDDLE_TIP, ext[2]),
        ("ring", handmodel.RING_MCP, handmodel.RING_PIP,
         handmodel.RING_DIP, handmodel.RING_TIP, ext[3]),
        ("little", handmodel.LITTLE_MCP, handmodel.LITTLE_PIP,
         handmodel.LITTLE_DIP, handmodel.LITTLE_TIP, ext[4]),
    )
    for name, mcp, pip, dip, tip, amount in chains:
        lo, hi = handmodel._CURL_RANGE[name]
        reach = (lo + (hi - lo) * amount) * palm
        # A curled finger folds back toward the palm; the classifier only reads
        # tip-to-knuckle distance, but bending it realistically keeps the drawn
        # skeleton honest too.
        angle = (1.0 - amount) * math.radians(150)
        dx, dy = math.sin(angle), -math.cos(angle)
        px[tip], py[tip] = px[mcp] + dx * reach, py[mcp] + dy * reach
        for joint, frac in ((pip, 0.42), (dip, 0.72)):
            px[joint] = px[mcp] + dx * reach * frac
            py[joint] = py[mcp] + dy * reach * frac

    lo, hi = handmodel._CURL_RANGE["thumb"]
    treach = (lo + (hi - lo) * ext[0]) * palm
    tlen = math.hypot(*thumb_dir) or 1.0
    tdx, tdy = thumb_dir[0] / tlen, thumb_dir[1] / tlen
    px[handmodel.THUMB_TIP] = px[handmodel.THUMB_MCP] + tdx * treach
    py[handmodel.THUMB_TIP] = py[handmodel.THUMB_MCP] + tdy * treach
    px[handmodel.THUMB_IP] = px[handmodel.THUMB_MCP] + tdx * treach * 0.55
    py[handmodel.THUMB_IP] = py[handmodel.THUMB_MCP] + tdy * treach * 0.55

    if pinch is not None:
        # Park the thumb tip a set fraction of a palm from the index tip, which
        # is exactly what the pinch metric measures.
        px[handmodel.THUMB_TIP] = px[handmodel.INDEX_TIP] + pinch * palm * 0.7
        py[handmodel.THUMB_TIP] = py[handmodel.INDEX_TIP] + pinch * palm * 0.7
        px[handmodel.THUMB_IP] = (px[handmodel.THUMB_MCP] + px[handmodel.THUMB_TIP]) / 2
        py[handmodel.THUMB_IP] = (py[handmodel.THUMB_MCP] + py[handmodel.THUMB_TIP]) / 2

    return Frame(hand=True, width=WIDTH, height=HEIGHT,
                 x=[v / ASPECT for v in px], y=py, conf=[conf] * 21)


CURLED = (0.02,) * 5


def pose(name: str) -> Frame:
    out = {
        "open_palm": make_hand((0.95,) * 5, thumb_dir=(-0.8, -0.5)),
        "fist": make_hand(CURLED, thumb_dir=(-0.9, 0.3)),
        "thumbs_up": make_hand((0.95, 0.02, 0.02, 0.02, 0.02), thumb_dir=(0.0, -1.0)),
        "thumbs_down": make_hand((0.95, 0.02, 0.02, 0.02, 0.02), thumb_dir=(0.0, 1.0)),
        "peace": make_hand((0.1, 0.95, 0.95, 0.02, 0.02), thumb_dir=(-0.9, 0.3)),
        "point": make_hand((0.1, 0.95, 0.02, 0.02, 0.02), thumb_dir=(-0.9, 0.3)),
        "three": make_hand((0.1, 0.95, 0.95, 0.95, 0.02), thumb_dir=(-0.9, 0.3)),
        "rock": make_hand((0.1, 0.95, 0.02, 0.02, 0.95), thumb_dir=(-0.9, 0.3)),
        "shaka": make_hand((0.95, 0.02, 0.02, 0.02, 0.95), thumb_dir=(-0.9, 0.3)),
        "ok_sign": make_hand((0.5, 0.9, 0.95, 0.95, 0.95), pinch=0.12),
        "pinch": make_hand((0.5, 0.9, 0.05, 0.05, 0.05), pinch=0.12),
    }[name]
    return out


def test_features() -> None:
    print("\nFeature extraction")
    h = handmodel.build(make_hand())
    check("open hand builds", h.ok)
    check("palm size is positive", h.palm > 0.01, f"palm={h.palm:.4f}")

    for amount in (0.0, 0.5, 1.0):
        h = handmodel.build(make_hand((amount,) * 5))
        recovered = h.ext[1]
        check(f"extension {amount:.1f} round-trips", abs(recovered - amount) < 0.08,
              f"got {recovered:.3f}")

    # Aspect correction is the whole reason this next one can pass: without it a
    # horizontal pinch and a vertical one of the same physical gap measure
    # nearly two to one apart on a 16:9 frame.
    near = handmodel.build(make_hand(pinch=0.10))
    far = handmodel.build(make_hand())
    check("pinch reads small when closed", near.pinch_index < 0.25,
          f"got {near.pinch_index:.3f}")
    check("pinch reads large when open", far.pinch_index > 0.6,
          f"got {far.pinch_index:.3f}")

    small = handmodel.build(make_hand(pinch=0.12, scale=0.10))
    large = handmodel.build(make_hand(pinch=0.12, scale=0.30))
    check("pinch is distance-invariant",
          abs(small.pinch_index - large.pinch_index) < 0.05,
          f"{small.pinch_index:.3f} vs {large.pinch_index:.3f}")

    low = handmodel.build(make_hand(conf=0.1), conf_min=0.35)
    check("low-confidence frames are rejected", not low.ok)


def test_poses() -> None:
    print("\nPose classification")
    for name in ("open_palm", "fist", "thumbs_up", "thumbs_down", "peace",
                 "point", "three", "rock", "shaka", "ok_sign", "pinch"):
        h = handmodel.build(pose(name))
        got = gestures.classify_pose(h)
        check(f"{name}", got == name, f"classified as {got}")

    # A hand with nothing distinctive going on must classify as nothing at all,
    # or every pose becomes reachable by accident on the way to another one.
    mush = handmodel.build(make_hand((0.45,) * 5, thumb_dir=(-0.9, 0.3)))
    check("ambiguous hand matches nothing", gestures.classify_pose(mush) is None,
          f"classified as {gestures.classify_pose(mush)}")


def test_gesture_engine() -> None:
    print("\nGesture timing")
    engine = gestures.GestureEngine()
    engine.bindings["thumbs_up"] = gestures.Binding(
        enabled=True, action="volume_up", mode="repeat", hold_ms=150, repeat_ms=100)
    hand = handmodel.build(pose("thumbs_up"))

    t = 1000.0
    fired = engine.update(hand, t)
    check("nothing fires on the first frame", not fired)

    t += 0.10
    check("nothing fires before the hold time", not engine.update(hand, t))
    t += 0.10
    check("fires once the hold time passes", len(engine.update(hand, t)) == 1)

    t += 0.05
    check("does not fire again immediately", not engine.update(hand, t))
    t += 0.10
    check("repeats after the repeat interval", len(engine.update(hand, t)) == 1)

    engine2 = gestures.GestureEngine()
    engine2.bindings["fist"] = gestures.Binding(enabled=True, action="media_play_pause",
                                                mode="once", hold_ms=100)
    fist = handmodel.build(pose("fist"))
    t = 2000.0
    engine2.update(fist, t)
    t += 0.2
    check("'once' fires", len(engine2.update(fist, t)) == 1)
    for _ in range(20):
        t += 0.1
        if engine2.update(fist, t):
            check("'once' does not repeat while held", False)
            break
    else:
        check("'once' does not repeat while held", True)


def test_swipes() -> None:
    print("\nSwipes")
    engine = gestures.GestureEngine()
    engine.bindings["swipe_right"] = gestures.Binding(enabled=True, action="media_next")
    engine.bindings["swipe_left"] = gestures.Binding(enabled=True, action="media_prev")

    def sweep(dx: float, flip_x: bool, dy: float = 0.0):
        eng = gestures.GestureEngine()
        for gid in ("swipe_right", "swipe_left", "swipe_up", "swipe_down"):
            eng.bindings[gid] = gestures.Binding(enabled=True, action="media_next")
        t = 5000.0
        out = []
        for i in range(16):
            frac = i / 15
            hand = handmodel.build(make_hand((0.95,) * 5, thumb_dir=(-0.8, -0.5),
                                             centre=(0.5 + dx * frac,
                                                     0.5 + dy * frac)))
            out += eng.update(hand, t, flip_x=flip_x)
            t += 0.012
        return [gid for gid, _ in out]

    check("moving right in frame is a left swipe when mirrored",
          "swipe_left" in sweep(0.22, True), str(sweep(0.22, True)))
    check("moving left in frame is a right swipe when mirrored",
          "swipe_right" in sweep(-0.22, True), str(sweep(-0.22, True)))
    check("a still hand is not a swipe", not sweep(0.0, True))

    # Small movements must not trigger, or the pointer's normal travel would fire
    # a swipe every time you reached for something.
    check("a small drift is not a swipe", not sweep(0.03, True), str(sweep(0.03, True)))

    # The same *physical* travel sideways and vertically must both register.
    # Before the palm centre was aspect-corrected, a horizontal swipe needed
    # 16:9 more distance than a vertical one to cross the same threshold.
    horizontal = 0.14                      # fraction of frame width
    vertical = horizontal * ASPECT         # the same distance, as height
    check("equal physical travel triggers on both axes",
          bool(sweep(horizontal, False)) and bool(sweep(0.0, False, vertical)),
          f"h={sweep(horizontal, False)} v={sweep(0.0, False, vertical)}")
    check("a diagonal favouring neither axis is not a swipe",
          not sweep(horizontal, False, vertical),
          str(sweep(horizontal, False, vertical)))


def test_filters() -> None:
    print("\nFilters")
    f = OneEuro2D(min_cutoff=1.0, beta=0.0)
    t = 0.0
    out = []
    for _ in range(200):
        t += 1 / 120
        out.append(f(100.0 + (1 if len(out) % 2 else -1), 100.0, t)[0])
    spread = max(out[50:]) - min(out[50:])
    check("noise on a still hand is smoothed away", spread < 0.6, f"spread={spread:.3f}")

    f2 = OneEuro2D(min_cutoff=1.0, beta=0.05)
    t, x = 0.0, 0.0
    for _ in range(120):
        t += 1 / 120
        x += 10.0
        got = f2(x, 0.0, t)[0]
    lag = x - got
    check("a fast move does not lag far behind", lag < 120, f"lag={lag:.1f}px")

    h = Hysteresis(0.30, 0.44)
    check("latches on below the low threshold", h.update(0.25))
    check("stays latched between thresholds", h.update(0.38))
    check("releases above the high threshold", not h.update(0.50))
    check("stays released between thresholds", not h.update(0.38))


def test_wire() -> None:
    print("\nWire format")
    values = []
    for i in range(21):
        values += [0.1 * i, 0.2, 0.9]
    payload = struct.pack("<IddBBBBHHf", 7, 1.0, 1.004, 1, 2, 1, 0, WIDTH, HEIGHT, 4.5)
    payload += struct.pack("<" + "fff" * 21, *values)
    packet = protocol.MAGIC + struct.pack("<BH", protocol.TYPE_FRAME, len(payload)) + payload

    reader = protocol.Reader()
    check("one packet decodes", len(reader.feed(packet, 0.0)) == 1)

    reader = protocol.Reader()
    got = []
    # Byte at a time: the parser has to survive TCP handing it a frame in
    # arbitrary slices, which is the normal case, not an edge case.
    for byte in packet:
        got += reader.feed(bytes([byte]), 0.0)
    check("survives being fed one byte at a time", len(got) == 1)
    if got:
        frame = got[0][1]
        check("fields survive the round trip",
              frame.seq == 7 and frame.width == WIDTH and frame.chirality == 2,
              f"seq={frame.seq} w={frame.width}")
        check("phone-side latency is derived", abs(frame.phone_ms - 4.0) < 0.001,
              f"{frame.phone_ms:.3f}")

    reader = protocol.Reader()
    check("resynchronises after junk",
          len(reader.feed(b"\x00\x01garbage" + packet, 0.0)) == 1)

    reader = protocol.Reader()
    check("two packets in one read", len(reader.feed(packet + packet, 0.0)) == 2)


def test_link() -> None:
    print("\nLink (against a fake phone)")
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    stop = threading.Event()

    def phone():
        conn, _ = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        seq = 0
        values = []
        for i in range(21):
            values += [0.5, 0.5, 0.9]
        body = struct.pack("<" + "fff" * 21, *values)
        while not stop.is_set():
            seq += 1
            head = struct.pack("<IddBBBBHHf", seq, time.time(), time.time() + 0.004,
                               1, 2, 1, 0, WIDTH, HEIGHT, 4.0)
            payload = head + body
            try:
                conn.sendall(protocol.MAGIC
                             + struct.pack("<BH", protocol.TYPE_FRAME, len(payload))
                             + payload)
            except OSError:
                break
            time.sleep(1 / 200)
        conn.close()

    threading.Thread(target=phone, daemon=True).start()

    received = []
    link = Link(lambda frame: received.append(frame))
    link.start("wifi", "127.0.0.1", port)
    time.sleep(1.2)
    count = len(received)
    link.stop()
    stop.set()
    server.close()

    check("frames arrive", count > 50, f"got {count} in 1.2 s")
    check("sequence numbers are in order",
          all(b.seq > a.seq for a, b in zip(received, received[1:])))
    check("phone latency is reported", received and received[0].phone_ms > 0)


def test_config_and_actions() -> None:
    print("\nConfig and actions")
    import json
    import tempfile

    from handpilot import actions, config

    cfg = config.load(path="does-not-exist.json")
    check("defaults load", cfg["pointer"]["anchor"] == "knuckles" and cfg["version"] == 2)
    check("every gesture has a binding",
          all(gid in cfg["gestures"]["bindings"] for gid in gestures.GESTURE_IDS),
          str(set(gestures.GESTURE_IDS) - set(cfg["gestures"]["bindings"])))
    check("every bound action exists",
          all(b["action"] in actions.ACTIONS
              for b in cfg["gestures"]["bindings"].values()))
    check("action labels round-trip",
          all(actions.id_for_label(actions.label_for(a)) == a
              for a in actions.ordered_ids()))
    check("there are plenty of actions", len(actions.ACTIONS) > 50,
          f"{len(actions.ACTIONS)}")

    # A file from the Tkinter version: its inverted "Smoothing" slider and its
    # lead-ahead must not survive the upgrade, and its region must be flagged.
    old = {"camera": {"fps": 240}, "pointer": {"min_cutoff": 5.0, "beta": 0.097, "lead_ms": 12.8,
                                               "pinch_on": 0.10, "pinch_off": 0.15,
                                               "region": [0.2, 0.1, 0.9, 0.8]}}
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(old, handle)
        migrated = config.load(path)
    p = migrated["pointer"]
    check("old filter values are not carried over",
          p["min_cutoff"] == config.DEFAULTS["pointer"]["min_cutoff"] and p["lead_ms"] == 0.0)
    check("old click thresholds start from the new defaults",
          p["pinch_on"] == config.DEFAULTS["pointer"]["pinch_on"])
    check("old region kept but flagged for recalibration",
          p["region"] == [0.2, 0.1, 0.9, 0.8] and p["needs_calibration"])
    check("camera settings left to the phone", "camera" not in migrated)


class _NoInput:
    """Swap out everything in winput that would touch the real mouse."""

    def __enter__(self):
        from handpilot import winput
        self.winput = winput
        self.saved = {name: getattr(winput, name) for name in
                      ("move_to", "button_down", "button_up", "wheel", "primary_screen",
                       "virtual_screen")}
        self.moves: list = []
        self.buttons: list = []
        winput.move_to = lambda x, y: self.moves.append((x, y))
        winput.button_down = lambda b="left": self.buttons.append(("down", b))
        winput.button_up = lambda b="left": self.buttons.append(("up", b))
        winput.wheel = lambda clicks: None
        winput.primary_screen = lambda: (0, 0, 1920, 1080)
        winput.virtual_screen = lambda: (0, 0, 1920, 1080)
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(self.winput, name, fn)


def _noisy(frame: Frame, rnd, tip_sigma: float, knuckle_sigma: float) -> Frame:
    xs, ys = [], []
    for i in range(21):
        s = tip_sigma if i in handmodel.TIPS else knuckle_sigma
        xs.append(frame.x[i] + rnd.gauss(0, s))
        ys.append(frame.y[i] + rnd.gauss(0, s * ASPECT))
    return Frame(hand=True, width=WIDTH, height=HEIGHT, x=xs, y=ys, conf=[0.9] * 21)


def test_pointer_pipeline() -> None:
    print("\nPointer pipeline (mouse output stubbed)")
    import random

    from handpilot.controller import PointerMode
    from handpilot.filters import Rope, feel

    steady = [feel(s) for s in (0.0, 0.5, 1.0)]
    check("steadiness lowers the cutoff and beta, widens the dead zone",
          steady[0][0] > steady[1][0] > steady[2][0] and steady[0][1] > steady[1][1] > steady[2][1]
          and steady[0][2] < steady[1][2] < steady[2][2])

    rope = Rope(2.0)
    rope(100.0, 100.0)
    held = {rope(100.0 + dx, 100.0) for dx in (0.5, -1.0, 1.5, -1.9)}
    check("the dead zone ignores movement inside its radius", held == {(100.0, 100.0)})
    x, _ = rope(110.0, 100.0)
    check("the dead zone follows movement beyond it", abs(x - 108.0) < 1e-9, f"{x}")

    open_hand = handmodel.build(make_hand((0.95,) * 5, thumb_dir=(-0.8, -0.5)))
    pinched = handmodel.build(make_hand((0.5, 0.9, 0.9, 0.9, 0.9), pinch=0.05))
    k_move = math.hypot(open_hand.knuckle_x - pinched.knuckle_x,
                        open_hand.knuckle_y - pinched.knuckle_y)
    p_move = math.hypot(open_hand.point_x - pinched.point_x, open_hand.point_y - pinched.point_y)
    check("pinching barely moves the knuckle tracking point",
          k_move < 0.25 * p_move, f"knuckles {k_move:.4f} vs pinch point {p_move:.4f}")

    def still_moves(cfg: dict) -> int:
        rnd = random.Random(7)
        base = make_hand((0.5, 0.9, 0.9, 0.9, 0.9), thumb_dir=(-0.8, -0.5))
        with _NoInput() as io:
            p = PointerMode()
            p.configure({"region": [0.2, 0.2, 0.8, 0.8], "click": False, **cfg})
            for n in range(360):
                p.update(handmodel.build(_noisy(base, rnd, 0.0012, 0.0004)), n / 120, n / 120)
            settled = io.moves[120:]
        return sum(1 for a, b in zip(settled, settled[1:]) if a != b)

    old = still_moves({"anchor": "pinch", "custom_filter": True, "min_cutoff": 5.0,
                       "beta": 0.097, "deadzone_px": 0.0, "lead_ms": 12.8})
    new = still_moves({"anchor": "knuckles", "steadiness": 0.5})
    check("a still hand holds the cursor still", new <= 2, f"moved {new} times in 2 s")
    check("...where the old settings shook it", old > 100, f"moved {old} times")

    # Frames that left the phone 8 ms apart but arrived in one USB read share a
    # receive time. The filter must still see two samples.
    with _NoInput() as io:
        p = PointerMode()
        p.configure({"region": [0.2, 0.2, 0.8, 0.8], "click": False,
                     "custom_filter": True, "min_cutoff": 50.0, "beta": 0.0, "deadzone_px": 0.0})
        for n, cx in enumerate((0.40, 0.40, 0.45, 0.50)):
            p.update(handmodel.build(make_hand(centre=(cx, 0.5))), 1.0, 10.0 + n / 120)
        xs = [x for x, _ in io.moves]
    check("filtering runs on capture time, not arrival time",
          len(set(xs[1:])) == len(xs[1:]), f"{xs}")


def test_raw_data() -> None:
    print("\nRaw data")
    import json

    from handpilot import HandStream, UdpForwarder

    frame = make_hand()
    frame.seq, frame.chirality = 5, 2
    row = frame.to_dict()
    check("frames serialise to the documented shape",
          set(row) == {"seq", "t", "hand", "handedness", "width", "height", "landmarks"}
          and len(row["landmarks"]) == 21 and row["handedness"] == "right")
    check("landmarks can be read by name",
          frame.landmark("index_tip")[:2] == (frame.x[8], frame.y[8]))

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    forwarder = UdpForwarder(f"127.0.0.1:{receiver.getsockname()[1]}")
    forwarder.send(frame)
    try:
        got = json.loads(receiver.recv(65536))
    except (socket.timeout, ValueError):
        got = {}
    forwarder.close()
    receiver.close()
    check("UDP output delivers one JSON frame per datagram", got.get("seq") == 5)

    port, stop = _fake_phone()
    with HandStream(wifi="127.0.0.1", port=port) as hands:
        first = hands.read(3.0)
    stop.set()
    check("HandStream yields frames without touching the mouse",
          first is not None and first.hand and len(first.x) == 21)


def _fake_phone(rate: float = 200.0):
    """A socket that speaks the phone's side of the wire format."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    stop = threading.Event()

    def serve():
        server.settimeout(5.0)
        try:
            conn, _ = server.accept()
        except OSError:
            return
        hand = make_hand()
        body = struct.pack("<" + "fff" * 21, *[v for i in range(21)
                                               for v in (hand.x[i], hand.y[i], 0.9)])
        seq = 0
        while not stop.is_set():
            seq += 1
            now = time.time()
            payload = struct.pack("<IddBBBBHHf", seq, now, now + 0.004, 1, 2, 2, 0,
                                  WIDTH, HEIGHT, 4.0) + body
            try:
                conn.sendall(protocol.MAGIC + struct.pack("<BH", protocol.TYPE_FRAME,
                                                          len(payload)) + payload)
            except OSError:
                break
            time.sleep(1 / rate)
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], stop


def test_app_server() -> None:
    print("\nApp window server")
    import json
    import tempfile
    import urllib.error
    import urllib.request

    from handpilot import config
    from handpilot.app import App
    from handpilot.server import UIServer

    port, stop = _fake_phone()
    cfg = config.load("does-not-exist.json")
    cfg.update(transport="wifi", wifi_host="127.0.0.1", port=port)
    with tempfile.TemporaryDirectory() as folder, _NoInput():
        app = App(cfg, config_path=os.path.join(folder, "config.json"))
        server = UIServer(app, 0, quit_on_close=False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        app.start(hotkeys=False)
        base = f"http://127.0.0.1:{server.port}"

        def call(path, body=None, token=True, host=None):
            headers = {"Content-Type": "application/json"}
            if token:
                headers["X-HandPilot-Token"] = server.token
            if host:
                headers["Host"] = host
            data = json.dumps(body).encode() if body is not None else None
            request = urllib.request.Request(base + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=3) as reply:
                    return reply.status, reply.read()
            except urllib.error.HTTPError as err:
                return err.code, err.read()

        status, page = call("/", token=False)
        check("serves the window", status == 200 and b"<canvas" in page)
        check("rejects API calls without the session token", call("/api/state", token=False)[0] == 401)
        check("rejects a foreign Host header (DNS rebinding)",
              call("/api/state", host="evil.example:80")[0] == 403)
        status, body = call("/api/state")
        check("serves state with the token", status == 200
              and json.loads(body)["config"]["pointer"]["anchor"] == "knuckles")

        status, _ = call("/api/config", {"path": "pointer.steadiness", "value": 0.8})
        check("applies a valid setting", status == 200
              and app.controller.pointer.rope.radius > 2.0)
        check("refuses unknown settings",
              call("/api/config", {"path": "pointer.nope", "value": 1})[0] == 400)
        check("refuses wrong types",
              call("/api/config", {"path": "pointer.click", "value": "yes"})[0] == 400)
        call("/api/config", {"path": "pointer.pinch_on", "value": 0.6})
        check("keeps the click hysteresis band open",
              cfg["pointer"]["pinch_off"] >= cfg["pointer"]["pinch_on"] + 0.04)

        deadline = time.time() + 3
        while time.time() < deadline and not app.controller.link.stats.connected:
            time.sleep(0.05)
        reply = urllib.request.urlopen(f"{base}/api/live?t={server.token}", timeout=3)
        line = reply.readline()
        reply.close()
        snap = json.loads(line[len(b"data: "):]) if line.startswith(b"data: ") else {}
        check("streams live snapshots", snap.get("conn", {}).get("connected") is True,
              line[:80].decode(errors="replace"))
        check("browser stream is off unless enabled", call("/stream", token=False)[0] == 403)

        app.quit_event.set()
        server.shutdown()
        server.server_close()
        app.shutdown()
    stop.set()


def test_pointer_math() -> None:
    print("\nPointer mapping")
    from handpilot.controller import PointerMode

    p = PointerMode()
    p.region = (0.2, 0.2, 0.8, 0.8)
    p.flip_x = True
    p.flip_y = False
    left, top, width, height = 0, 0, 1920, 1080
    try:
        left, top, width, height = p.screen_rect()
    except OSError:
        pass

    x_left, _ = p._to_screen(0.2, 0.5)
    x_right, _ = p._to_screen(0.8, 0.5)
    check("X is mirrored", x_left > x_right, f"{x_left:.0f} vs {x_right:.0f}")
    check("region edges reach the screen edges",
          abs(x_right - left) < 2 and abs(x_left - (left + width - 1)) < 2,
          f"{x_left:.0f}..{x_right:.0f} for {left}..{left + width - 1}")

    _, y_top = p._to_screen(0.5, 0.2)
    _, y_bottom = p._to_screen(0.5, 0.8)
    check("Y is not mirrored", y_top < y_bottom)

    out_x, _ = p._to_screen(1.5, 0.5)
    check("outside the region clamps", left <= out_x <= left + width)


def main() -> int:
    print("HandPilot self-test (no phone required)")
    test_features()
    test_poses()
    test_gesture_engine()
    test_swipes()
    test_filters()
    test_wire()
    test_link()
    test_config_and_actions()
    test_pointer_math()
    test_pointer_pipeline()
    test_raw_data()
    test_app_server()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

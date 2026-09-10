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
    from handpilot import actions, config

    cfg = config.load(path="does-not-exist.json")
    check("defaults load", cfg["camera"]["fps"] == 120)
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


def test_pointer_math() -> None:
    print("\nPointer mapping")
    from handpilot.controller import PointerMode

    p = PointerMode()
    p.region = (0.2, 0.2, 0.8, 0.8)
    p.flip_x = True
    p.flip_y = False
    left, top, width, height = 0, 0, 1920, 1080
    try:
        from handpilot import winput
        left, top, width, height = winput.virtual_screen()
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
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

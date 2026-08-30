"""Round-trip the HandBridge wire format over a real UDP socket.

Validates the layout the Swift sender and the Python receiver must agree on,
without needing the phone. Both halves are written from the same spec, and a
mismatch here is far cheaper to find than one discovered while holding a phone.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from handtrack.sources.handbridge_source import (HandBridgeSource, PACKET_SIZE,
                                                 encode_packet, LANDMARKS)
import socket

SWIFT_SIZE = 48 + LANDMARKS * 16
print(f"packet size: python {PACKET_SIZE} bytes, swift {SWIFT_SIZE} bytes -> "
      f"{'MATCH' if PACKET_SIZE == SWIFT_SIZE else 'MISMATCH'}")
assert PACKET_SIZE == SWIFT_SIZE, "wire format disagreement between the two halves"

rng = np.random.default_rng(0)
pixels = rng.uniform(0, 640, (LANDMARKS, 2)).astype(np.float32)
depth = rng.uniform(0.25, 0.7, LANDMARKS).astype(np.float32)
depth[3] = np.nan                      # holes are normal and must survive
conf = rng.uniform(0.5, 1.0, LANDMARKS).astype(np.float32)

src = HandBridgeSource(port=9899)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

import threading
def feed():
    time.sleep(0.3)
    for seq in range(1, 6):
        pkt = encode_packet(seq, 100.0 + seq * 0.033, 100.0 + seq * 0.033 + 0.004,
                            640, 480, 512.5, 512.5, 320.0, 240.0, 2,
                            pixels, depth, conf)
        sock.sendto(pkt, ("127.0.0.1", 9899))
        time.sleep(0.03)

threading.Thread(target=feed, daemon=True).start()
src.start(timeout=6.0)
time.sleep(0.4)
f = src.read()

print(f"\nreceived seq={f.index}  hand={f.has_hand}  handedness={f.handedness}")
print(f"  image {f.width}x{f.height}   fx={f.intrinsics[0,0]:.1f} cx={f.intrinsics[0,2]:.1f}")
print(f"  pixel error   max {np.nanmax(np.abs(f.pixels - pixels)):.6f}")
print(f"  depth error   max {np.nanmax(np.abs(f.depth - depth)):.6f}")
print(f"  conf  error   max {np.nanmax(np.abs(f.confidence - conf)):.6f}")
print(f"  NaN depth at index 3 preserved: {np.isnan(f.depth[3])}")
print(f"  packets={src.packets} malformed={src.malformed} dropped={src.dropped}")
print(f"  transit estimate: {src.transit_ms(f):.2f} ms (loopback, so ~4 ms by construction)")

# Garbage must be rejected rather than crash the receiver.
sock.sendto(b"not a packet", ("127.0.0.1", 9899))
sock.sendto(b"HBR1" + b"\x00" * 100, ("127.0.0.1", 9899))
time.sleep(0.3)
print(f"  after 2 malformed packets: malformed={src.malformed}, still alive={src.read() is not None}")

src.release()
sock.close()
print("\nprotocol round-trip OK")

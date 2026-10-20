"""Receive HandPilot's UDP output: one JSON frame per datagram.

In the HandPilot window, switch on Raw data > UDP (default target
127.0.0.1:9872), then run this. Any language that can read UDP can do the same.
"""

import json
import socket

PORT = 9872

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", PORT))
print(f"listening on udp://127.0.0.1:{PORT} (Ctrl+C to stop)")

try:
    while True:
        frame = json.loads(sock.recv(65536))
        if not frame["hand"]:
            continue
        wrist = frame["landmarks"][0]
        index_tip = frame["landmarks"][8]
        print(f"frame {frame['seq']:>7}   wrist {wrist[0]:.3f},{wrist[1]:.3f}   "
              f"index tip {index_tip[0]:.3f},{index_tip[1]:.3f}", end="\r")
except KeyboardInterrupt:
    pass

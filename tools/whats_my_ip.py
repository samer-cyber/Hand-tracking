"""Print the LAN address to type into the HandBridge app on the phone."""
import socket

def lan_ip() -> str:
    # Connecting a UDP socket sends nothing, but makes the OS pick the interface
    # it would actually use to reach the outside world -- which is the one the
    # phone can see, unlike whatever gethostbyname happens to return.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

if __name__ == "__main__":
    ip = lan_ip()
    print(f"\n  Enter this in the HandBridge app on your phone:\n")
    print(f"      PC address:  {ip}")
    print(f"      Port:        9870\n")
    print("  Both devices must be on the same Wi-Fi network.")
    print("  If no packets arrive, allow Python through the Windows firewall")
    print("  for Private networks (inbound UDP).\n")

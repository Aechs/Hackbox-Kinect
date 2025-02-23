import socket
import os
import fcntl
import struct
import math

# Define the framebuffer device
FB_DEVICE = "/dev/fb0"

# Open the framebuffer device
fb = os.open(FB_DEVICE, os.O_RDWR)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5000))

print("Listening for UDP traffic on port 5000...")

try:
    while True:
        # Receive data from the socket
        data, addr = sock.recv(540)
        
        # Write the data to the framebuffer
        os.write(fb, data)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Close the framebuffer device
    os.close(fb)
    # Close the socket
    sock.close()
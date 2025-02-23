import socket
import time

# Define the target IP address and port
UDP_IP = "192.168.0.209"
UDP_PORT = 5000

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Function to send framebuffer data
def send_framebuffer_data(data):
    sock.sendto(data, (UDP_IP, UDP_PORT))

# Example framebuffer data (replace with actual data)
framebuffer_data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09'

# Send framebuffer data in a loop
while True:
    send_framebuffer_data(framebuffer_data)
    time.sleep(10)  # Adjust the sleep time as needed
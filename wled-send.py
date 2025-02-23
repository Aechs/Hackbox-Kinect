import struct
import socket


# Define the UDP host and port
UDP_IP = "129.21.71.59"
UDP_PORT = 11988

# #define UDP_SYNC_HEADER_V2 "00002"
# // new "V2" AC 0.14.0 audiosync struct - 40 Bytes
# struct audioSyncPacket_v2 {
#       char    header[6] = UDP_SYNC_HEADER_V2; // 06 bytes, last byte is '\0' as string terminator.
#       float   sampleRaw;      //  04 Bytes  - either "sampleRaw" or "rawSampleAgc" depending on soundAgc setting
#       float   sampleSmth;     //  04 Bytes  - either "sampleAvg" or "sampleAgc" depending on soundAgc setting
#       uint8_t samplePeak;     //  01 Bytes  - 0 no peak; >=1 peak detected. In future, this will also provide peak Magnitude
#       uint8_t reserved1;      //  01 Bytes  - reserved for future extensions like loudness
#       uint8_t fftResult[16];  //  16 Bytes  - FFT results, one byte per GEQ channel
#       float  FFT_Magnitude;   //  04 Bytes  - magnitude of strongest peak in FFT
#       float  FFT_MajorPeak;   //  04 Bytes  - frequency of strongest peak in FFT
# };

# Define the data to be sent
header = b"00002\0"
sampleRaw = 0.0
sampleSmth = 0.0
samplePeak = 0
reserved1 = 0
fftResult = [0] * 16
FFT_Magnitude = 0.0
FFT_MajorPeak = 0.0

# Pack the data into a binary format
data = struct.pack(
    '6sf f B B 16B f f',
    header,
    sampleRaw,
    sampleSmth,
    samplePeak,
    reserved1,
    *fftResult,
    FFT_Magnitude,
    FFT_MajorPeak
)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send the data packet to the UDP host
sock.sendto(data, (UDP_IP, UDP_PORT))

print("Data packet sent to {}:{}".format(UDP_IP, UDP_PORT))
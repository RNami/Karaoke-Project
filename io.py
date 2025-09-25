import pyaudio
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

try:
    while True:
        data = stream.read(CHUNK)
        print(f"Received chunk of {len(data)} bytes")
        # Process 'data' here, for example, send to a speech recognition API
except KeyboardInterrupt:
    pass

print("Finished recording.")

stream.stop_stream()
stream.close()
p.terminate()

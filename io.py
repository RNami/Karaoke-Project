import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_FILENAME = "five_seconds.wav"

RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
frames = []

print(f"Recording for {RECORD_SECONDS} seconds...")
for _ in range(num_chunks):
    data = stream.read(CHUNK)
    frames.append(data)
print(f"Received {RECORD_SECONDS} seconds of audio.")

print(f"Saving to {OUTPUT_FILENAME}...")
wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

stream.stop_stream()
stream.close()
p.terminate()
print("Done.")
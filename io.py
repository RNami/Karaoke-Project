import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
OUTPUT_FILENAME = "five_seconds.wav"
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

print("Available audio input devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"  Index {i}: {info['name']}")

USB_MIC_INDEX = int(input("Enter the device index for your USB microphone: "))

info = p.get_device_info_by_index(USB_MIC_INDEX)
print(f"Device info: {info}")

CHANNELS = info['maxInputChannels']
RATE = int(info['defaultSampleRate'])

print('Channels:', CHANNELS)
print('Rate:', RATE)

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=USB_MIC_INDEX,
                    frames_per_buffer=CHUNK,
                    output=False,)
except Exception as e:
    print(f"Could not open stream: {e}")
    p.terminate()
    exit(1)

num_chunks = int(RATE / CHUNK * RECORD_SECONDS)
print(f"Expecting to read {num_chunks} chunks.")
frames = []

print(f"Recording for {RECORD_SECONDS} seconds...")
for _ in range(num_chunks):
    data = stream.read(CHUNK, exception_on_overflow=False)
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



"""
List all available audio input/output devices with PyAudio.

Usage:
    python list_audio_devices.py
"""

import pyaudio

def list_devices():
    pa = pyaudio.PyAudio()
    device_count = pa.get_device_count()

    print("=== Available Audio Devices ===")
    for i in range(device_count):
        info = pa.get_device_info_by_index(i)
        name = info.get("name")
        max_input = info.get("maxInputChannels")
        max_output = info.get("maxOutputChannels")
        rate = int(info.get("defaultSampleRate"))

        print(f"[{i}] {name}")
        print(f"   Max Input Channels : {max_input}")
        print(f"   Max Output Channels: {max_output}")
        print(f"   Default SampleRate : {rate}")
        print("-" * 40)

    pa.terminate()

if __name__ == "__main__":
    list_devices()

"""
Simplified interactive PyAudio device chooser:
- Only shows devices from Windows WASAPI host API.
- Only lists: default microphone (input), default speaker (output), loopback (input).
- Lets user pick input/output easily.
"""
import os
import time

import numpy as np
import pyaudio

FORMAT = pyaudio.paInt16
COMMON_SAMPLE_RATES = [48000, 44100]
BUFFER_SIZES = [512, 1024]
PREFERRED_CHANNELS = [1, 2]


def get_wasapi_devices(pa):
    """Return only WASAPI devices (mic, speaker, loopback)."""
    devices = []
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        hostapi_name = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        if "WASAPI" not in hostapi_name and os.name == 'nt':
            continue
        devices.append({
            "index": idx,
            "name": info["name"],
            "max_in": int(info["maxInputChannels"]),
            "max_out": int(info["maxOutputChannels"]),
            "default_sr": int(info["defaultSampleRate"]),
        })
    return devices


def print_devices(devs):
    print("=== Simplified Device List (WASAPI only) ===")
    for d in devs:
        kind = []
        if d["max_in"] > 0:
            kind.append("INPUT")
        if d["max_out"] > 0:
            kind.append("OUTPUT")
        print(f"[{d['index']}] {d['name']} | {','.join(kind)} | defSR={d['default_sr']}")
    print("===========================================")


def prompt_device(devs, want_input=True):
    """Ask user to pick input or output device index."""
    while True:
        idx = input(f"Enter {'INPUT' if want_input else 'OUTPUT'} device index: ").strip()
        try:
            idx = int(idx)
        except ValueError:
            print("Please enter a number.")
            continue
        dev = next((d for d in devs if d["index"] == idx), None)
        if dev is None:
            print("Not found. Try again.")
            continue
        if want_input and dev["max_in"] < 1:
            print("This is not an input device.")
            continue
        if not want_input and dev["max_out"] < 1:
            print("This is not an output device.")
            continue
        return dev


def try_open(pa, inp, out, callback):
    """Try open stream with safe defaults (48k/44.1k mono, 512/1024 buffer)."""
    for rate in COMMON_SAMPLE_RATES:
        for ch in PREFERRED_CHANNELS:
            for buf in BUFFER_SIZES:
                try:
                    stream = pa.open(
                        format=FORMAT,
                        channels=ch,
                        rate=rate,
                        input=True,
                        output=True,
                        frames_per_buffer=buf,
                        input_device_index=inp["index"],
                        output_device_index=out["index"],
                        stream_callback=callback,
                    )
                    return stream, rate, ch, buf
                except Exception:
                    continue
    raise RuntimeError("Could not open any valid stream.")


def main():
    pa = pyaudio.PyAudio()
    try:
        devs = get_wasapi_devices(pa)
        if not devs:
            print("No WASAPI devices found.")
            return
        print_devices(devs)

        inp = prompt_device(devs, want_input=True)
        out = prompt_device(devs, want_input=False)

        print(f"\nUsing INPUT : [{inp['index']}] {inp['name']}")
        print(f"Using OUTPUT: [{out['index']}] {out['name']}\n")

        def callback(in_data, frame_count, time_info, status):
            x = np.frombuffer(in_data, dtype=np.int16)
            y = x  # passthrough, insert DSP here
            return (y.tobytes(), pyaudio.paContinue)

        stream, rate, ch, buf = try_open(pa, inp, out, callback)
        print(f"✅ Stream opened: rate={rate}, ch={ch}, buffer={buf}")
        print("Streaming... Ctrl+C to stop.")

        stream.start_stream()
        while stream.is_active():
            print('buf:', buf)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        pa.terminate()
        print("Audio terminated.")


if __name__ == "__main__":
    main()

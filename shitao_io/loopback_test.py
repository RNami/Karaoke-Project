"""
loop_back_test.py

Interactive mic→speaker monitor for Windows:
- Lists WASAPI devices.
- Lets user pick *one* INPUT (mic) and *one* OUTPUT (speaker).
- Opens two independent PyAudio streams (input-only & output-only).
- If sample rates differ, resamples with scipy.signal.resample_poly.
"""

import os
import time
import numpy as np
import pyaudio
from scipy.signal import resample_poly, lfilter

FORMAT = pyaudio.paInt16
BUFFER_SIZE = 1024

# === Audio effects ============================================================
def robot_voice_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
    freq = 80  # Hz
    t = np.arange(len(buffer)) / rate
    square_wave = np.sign(np.sin(2 * np.pi * freq * t))
    effected = buffer * square_wave
    return effected.astype(np.int16)


def concert_hall_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
    delay_ms = 100
    decay = 0.4
    delay_samples = int(rate * delay_ms / 1000)
    b = np.zeros(delay_samples + 1)
    b[0] = 1
    a = np.zeros(delay_samples + 1)
    a[0] = 1
    a[-1] = -decay
    effected = lfilter(b, a, buffer)
    effected = np.clip(effected, -32768, 32767)
    return effected.astype(np.int16)


def get_wasapi_devices(pa):
    """Return only WASAPI devices (mic, speaker, loopback)."""
    devs = []
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        hostapi = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        if "WASAPI" not in hostapi and os.name == "nt":
            continue
        devs.append({
            "index": idx,
            "name": info["name"],
            "max_in": int(info["maxInputChannels"]),
            "max_out": int(info["maxOutputChannels"]),
            "default_sr": int(info["defaultSampleRate"]),
        })
    return devs


def print_devices(devs):
    print("=== WASAPI Devices ===")
    for d in devs:
        kinds = []
        if d["max_in"] > 0:
            kinds.append("INPUT")
        if d["max_out"] > 0:
            kinds.append("OUTPUT")
        print(f"[{d['index']}] {d['name']} | {','.join(kinds)} | defSR={d['default_sr']}")
    print("=======================")


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
        if not dev:
            print("Not found.")
            continue
        if want_input and dev["max_in"] < 1:
            print("Not an input device.")
            continue
        if not want_input and dev["max_out"] < 1:
            print("Not an output device.")
            continue
        return dev


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

        in_rate = int(inp["default_sr"])
        out_rate = int(out["default_sr"])
        in_channels = min(inp["max_in"], 2)
        out_channels = min(out["max_out"], 2)

        print(f"Opening mic stream @ {in_rate} Hz, {in_channels} ch")
        in_stream = pa.open(format=FORMAT,
                            channels=in_channels,
                            rate=in_rate,
                            input=True,
                            frames_per_buffer=BUFFER_SIZE,
                            input_device_index=inp["index"])

        print(f"Opening speaker stream @ {out_rate} Hz, {out_channels} ch")
        out_stream = pa.open(format=FORMAT,
                             channels=out_channels,
                             rate=out_rate,
                             output=True,
                             frames_per_buffer=BUFFER_SIZE,
                             output_device_index=out["index"])

        print("Streams opened. Ctrl+C to stop.")
        ratio = out_rate / in_rate

        while True:
            data = in_stream.read(BUFFER_SIZE, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.int16)

            # If mic and speaker have different rates, resample
            if in_rate != out_rate:
                x = resample_poly(x, out_rate, in_rate)

            # Simple channel handling: duplicate to match output channels if needed
            if out_channels > in_channels:
                x = np.repeat(x.reshape(-1, in_channels), out_channels, axis=1)
            elif out_channels == 1 and x.ndim > 1:
                x = x.mean(axis=1)

            y = concert_hall_effect(x, out_rate)
            # y = x
            out_stream.write(y.astype(np.int16).tobytes())

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            in_stream.stop_stream(); in_stream.close()
            out_stream.stop_stream(); out_stream.close()
        except Exception:
            pass
        pa.terminate()
        print("Audio terminated.")


if __name__ == "__main__":
    main()

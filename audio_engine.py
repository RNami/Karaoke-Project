# audio_engine.py
import os

import pyaudio
import numpy as np
from scipy.signal import lfilter, resample_poly
import threading

FORMAT = pyaudio.paInt16
BUFFER_SIZE = 1024   # fixed internal buffer, GUI can still show value if desired


def get_wasapi_devices(pa):
    """Return only WASAPI devices (mic / speaker / loopback)."""
    devices = []
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        hostapi_name = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        if "WASAPI" not in hostapi_name and os.name == "nt":
            continue
        devices.append({
            "index": idx,
            "name": info["name"],
            "max_in": int(info["maxInputChannels"]),
            "max_out": int(info["maxOutputChannels"]),
            "default_sr": int(info["defaultSampleRate"]),
        })
    return devices


# === Audio effects ============================================================
def robot_voice_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
    freq = 80  # Hz
    t = np.arange(len(buffer)) / rate
    square_wave = np.sign(np.sin(2 * np.pi * freq * t))
    effected = buffer * square_wave
    return effected.astype(np.int16)


def concert_hall_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
    delay_ms = 300
    decay = 0.1
    delay_samples = int(rate * delay_ms / 1000)
    b = np.zeros(delay_samples + 1)
    b[0] = 1
    a = np.zeros(delay_samples + 1)
    a[0] = 1
    a[-1] = -decay
    effected = lfilter(b, a, buffer)
    effected = np.clip(effected, -32768, 32767)
    return effected.astype(np.int16)


# === Engine ===================================================================
class AudioEngine:
    """
    Opens TWO independent streams:
       mic → read at mic rate
       speaker → write at speaker rate
    Handles resampling + effect processing in Python.
    """
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.in_stream = None
        self.out_stream = None
        self.running = False
        self.effect_name = "None"
        self.current_level = 0.0  # mic RMS level 0–100
        self.lock = threading.Lock()

    def start_stream(self, inp_idx, out_idx, effect_name="None"):
        # Query device defaults
        in_dev = self.pa.get_device_info_by_index(inp_idx)
        out_dev = self.pa.get_device_info_by_index(out_idx)
        self.in_rate = int(in_dev["defaultSampleRate"])
        self.out_rate = int(out_dev["defaultSampleRate"])
        self.in_channels = max(1, min(in_dev["maxInputChannels"], 2))
        self.out_channels = max(1, min(out_dev["maxOutputChannels"], 2))
        self.effect_name = effect_name

        # Input-only stream
        self.in_stream = self.pa.open(format=FORMAT,
                                      channels=self.in_channels,
                                      rate=self.in_rate,
                                      input=True,
                                      frames_per_buffer=BUFFER_SIZE,
                                      input_device_index=inp_idx)

        # Output-only stream
        self.out_stream = self.pa.open(format=FORMAT,
                                       channels=self.out_channels,
                                       rate=self.out_rate,
                                       output=True,
                                       frames_per_buffer=BUFFER_SIZE,
                                       output_device_index=out_idx)

        self.running = True

        # Start worker thread
        import threading
        self.thread = threading.Thread(target=self._io_loop, daemon=True)
        self.thread.start()

    # -------------------------------------------------------------------------
    def _io_loop(self):
        """Read from mic, apply effect, resample, write to speakers."""
        while True:
            with self.lock:
                if not self.running or self.in_stream is None or self.out_stream is None:
                    break
                try:
                    data = self.in_stream.read(BUFFER_SIZE, exception_on_overflow=False)
                except Exception:
                    break

            x = np.frombuffer(data, dtype=np.int16)

            # RMS level
            self.current_level = min(100.0, (np.sqrt(np.mean(x.astype(np.float32)**2)) / 32768.0) * 100.0)

            # Effect
            if self.effect_name == "Robot Voice":
                x = robot_voice_effect(x, self.in_rate)
            elif self.effect_name == "Concert Hall":
                x = concert_hall_effect(x, self.in_rate)

            # Resample if needed
            if self.in_rate != self.out_rate:
                x = resample_poly(x, self.out_rate, self.in_rate)

            # Channel match
            if self.out_channels > self.in_channels:
                x = np.repeat(x.reshape(-1, self.in_channels), self.out_channels, axis=1)
            elif self.out_channels == 1 and x.ndim > 1:
                x = x.mean(axis=1)

            # write safely
            with self.lock:
                if self.running and self.out_stream is not None:
                    try:
                        self.out_stream.write(x.astype(np.int16).tobytes())
                    except OSError:
                        break

    # -------------------------------------------------------------------------
    def stop_stream(self):
        with self.lock:
            if not self.running:
                return 
            self.running = False

            if self.in_stream:
                try:
                    self.in_stream.stop_stream()
                    self.in_stream.close()
                except Exception:
                    pass
                self.in_stream = None

            if self.out_stream:
                try:
                    self.out_stream.stop_stream()
                    self.out_stream.close()
                except Exception:
                    pass
                self.out_stream = None

        # wait for worker thread to finish outside the lock
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)

    # -------------------------------------------------------------------------

    def terminate(self):
        self.stop_stream()
        self.pa.terminate()


    @property
    def stream(self):
        return self.in_stream

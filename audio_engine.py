import pyaudio
import numpy as np
from scipy.signal import lfilter

FORMAT = pyaudio.paInt16
COMMON_SAMPLE_RATES = [48000, 44100]
BUFFER_SIZES = [512, 1024]
PREFERRED_CHANNELS = [1, 2]

def get_wasapi_devices(pa):
    devices = []
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        hostapi_name = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        if "WASAPI" not in hostapi_name:
            continue
        devices.append({
            "index": idx,
            "name": info["name"],
            "max_in": int(info["maxInputChannels"]),
            "max_out": int(info["maxOutputChannels"]),
            "default_sr": int(info["defaultSampleRate"]),
        })
    return devices

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

class AudioEngine:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    def start_stream(self, inp_idx, out_idx, rate, ch, buf, effect_name):
        def callback(in_data, frame_count, time_info, status):
            x = np.frombuffer(in_data, dtype=np.int16)
            if effect_name == "Robot Voice":
                y = robot_voice_effect(x, rate)
            elif effect_name == "Concert Hall":
                y = concert_hall_effect(x, rate)
            else:
                y = x
            return (y.tobytes(), pyaudio.paContinue)

        self.stream = self.pa.open(
            format=FORMAT,
            channels=ch,
            rate=rate,
            input=True,
            output=True,
            frames_per_buffer=buf,
            input_device_index=inp_idx,
            output_device_index=out_idx,
            stream_callback=callback,
        )
        self.running = True
        self.stream.start_stream()

    def stop_stream(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def terminate(self):
        self.stop_stream()
        self.pa.terminate()
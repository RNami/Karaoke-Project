# input_output_real.py
import numpy as np
import pyaudio
from in_out import BLOCK, list_devices, choose_device_index

# int16 <-> float32 helpers
_INT16_SCALE = 32768.0

def int16_to_float32(data_bytes, channels):
    x = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32)
    if channels > 1:
        x = x.reshape(-1, channels)
    else:
        x = x.reshape(-1, 1)
    return x / _INT16_SCALE

def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * _INT16_SCALE).astype(np.int16).tobytes()

def open_duplex_stream(convolver_cb,
                       in_dev=None, out_dev=None, fs=None,
                       out_channels=1):
    """
    Opens a full-duplex callback stream:
      - input: mono (1 ch) to avoid invalid channel errors
      - output: out_channels (from IR), int16
      - frames_per_buffer = BLOCK
    convolver_cb(frames_float32) -> frames_float32_out
    """
    p = pyaudio.PyAudio()
    inputs, outputs = list_devices(p)

    # choose devices if not forced
    if in_dev is None:
        in_dev = choose_device_index(inputs, "Available audio input devices:")
    if out_dev is None:
        out_dev = choose_device_index(outputs, "Available audio output devices:")

    # pick default rates from devices if fs not forced
    if fs is None:
        if in_dev is not None:
            fs = int(p.get_device_info_by_index(in_dev)["defaultSampleRate"])
        else:
            fs = int(p.get_default_input_device_info()["defaultSampleRate"])

    FORMAT = pyaudio.paInt16
    IN_CH  = 1  # force mono mic to avoid [Errno -9998] Invalid number of channels

    def callback(in_data, frame_count, time_info, status):
        # Convert input bytes -> float32 mono frames
        x = int16_to_float32(in_data, IN_CH)  # shape: (BLOCK, 1)
        y = convolver_cb(x)                   # shape: (BLOCK, out_channels) float32
        out_bytes = float32_to_int16(y)
        return (out_bytes, pyaudio.paContinue)

    stream = p.open(format=FORMAT,
                    channels=out_channels,          # output channels (from IR)
                    rate=int(fs),
                    input=True,
                    output=True,
                    input_device_index=in_dev,
                    output_device_index=out_dev,
                    frames_per_buffer=BLOCK,
                    stream_callback=callback)
    return p, stream, fs

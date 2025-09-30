# Filters/ir_utils.py
import os, math
from typing import Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
import soundfile as sf

# === Minimal IR loader & FDL convolver =======================================
COMMON_IR_KEYS = ["h_air", "ir", "h", "rir", "impulse_response", "IR", "H"]
COMMON_INFO_KEYS = ["air_info", "info", "AIR_info"]



def load_ir_wav(ir_path: str) -> Tuple[np.ndarray, int]:
    x, fs = sf.read(ir_path)
    x = x[:, None] if x.ndim == 1 else x                # To ensure seamless compatibility with mono and multichannel inputs
    return x.astype(np.float32), fs

def load_ir_mat(ir_path: str) -> Tuple[np.ndarray, int]:
    data = loadmat(ir_path)
    x = data["h_air"].astype(np.float32)
    fs = int(data["air_info"]["fs"])

    return x.T, fs

loading_functions = {
    ".wav": load_ir_wav,
    ".mat": load_ir_mat
}


def load_ir_any(ir_path: str) -> Tuple[np.ndarray, int]:
    """
    Loads impulse response and the sampling rate from a wav file.
    :param ir_path: Relative path to the impulse response.
    :return: impulse response as np.float32 and sampling rate as an integer.
    """
    ext = os.path.splitext(ir_path)[1].lower()

    if ext not in loading_functions:
        raise ValueError(f"Unsupported file format: {ext}")
    return loading_functions[ext](ir_path)


def resample_if_needed(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    g = math.gcd(fs_out, fs_in)
    up, down = fs_out // g, fs_in // g
    channels = [resample_poly(x[:, c], up, down).astype(np.float32) for c in range(x.shape[1])]
    minimum_length_of_channels = min(len(channel) for channel in channels)
    return np.stack([ch[:minimum_length_of_channels] for ch in channels], axis=1)

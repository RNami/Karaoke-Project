# realtime_conv.py
import sys, glob
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly
from numpy.fft import rfft, irfft

from in_out import BLOCK
from input_output_real import open_duplex_stream

# --------- IR loader (minimal) ----------
COMMON_IR_KEYS = ["h_air", "ir", "h", "rir", "impulse_response", "IR", "H"]
COMMON_INFO_KEYS = ["air_info", "info", "AIR_info"]

def _extract_scalar(v):
    if isinstance(v, np.ndarray):
        v = np.squeeze(v)
        if v.size == 1: return float(v)
    try: return float(v)
    except Exception: return None

def _get_struct_field(struct_obj, name: str):
    if hasattr(struct_obj, "dtype") and struct_obj.dtype is not None and struct_obj.dtype.names:
        try: field = struct_obj[name]
        except Exception: return None
        return np.squeeze(field)
    return None

def _pick_ir_array(d: dict) -> np.ndarray:
    for k in COMMON_IR_KEYS:
        if k in d and isinstance(d[k], np.ndarray):
            return np.atleast_2d(d[k]).astype(np.float32)
    # fallback: largest numeric array
    cands = []
    for k, v in d.items():
        if k.startswith("__"): continue
        if isinstance(v, np.ndarray) and v.ndim in (1, 2) and np.issubdtype(v.dtype, np.number):
            cands.append((k, v))
    if not cands:
        raise RuntimeError("No numeric IR array found in .mat file.")
    _, v = max(cands, key=lambda kv: kv[1].size)
    return np.atleast_2d(v).astype(np.float32)

def _find_fs_in_info_struct(d: dict) -> Optional[float]:
    for info_key in COMMON_INFO_KEYS:
        if info_key in d:
            info = d[info_key]
            for cand in ["fs", "Fs", "FS", "sampling_rate", "sr", "sample_rate", "samplingRate"]:
                field = _get_struct_field(info, cand)
                if field is None: continue
                val = _extract_scalar(field)
                if val and val > 0: return float(val)
    return None

def load_ir_any(ir_path: str) -> Tuple[np.ndarray, int]:
    p = Path(ir_path)
    if not p.exists():
        raise FileNotFoundError(ir_path)
    if p.suffix.lower() == ".mat":
            d = loadmat(str(p), squeeze_me=True)
            ir = _pick_ir_array(d)
            if ir.ndim == 1: ir = ir[:, None]
            if ir.ndim == 2 and ir.shape[0] < 8 and ir.shape[1] > 8:
                ir = ir.T
            fs = _find_fs_in_info_struct(d)
            if fs is None:
                for cand in ["fs", "Fs", "FS", "sampling_rate", "sr", "sample_rate", "samplingRate"]:
                    if cand in d:
                        fs = _extract_scalar(d[cand]); 
                        if fs: break
            if fs is None:
                raise RuntimeError("IR sample rate not found in .mat")
            return ir.astype(np.float32), int(round(fs))
    elif p.suffix.lower() == ".wav":
        ir, fs = sf.read(str(p), always_2d=True)
        return ir.astype(np.float32), int(round(fs))
    else:
        # allow a directory (pick first .mat/.wav)
        if p.is_dir():
            mats = sorted(glob.glob(str(p / "*.mat")))
            if mats:
                return load_ir_any(mats[0])
            wavs = sorted(glob.glob(str(p / "*.wav")))
            if wavs:
                return load_ir_any(wavs[0])
        raise RuntimeError(f"Unsupported IR path: {ir_path}")

def resample_if_needed(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out: return x
    from math import gcd
    g = gcd(fs_out, fs_in)
    up, down = fs_out // g, fs_in // g
    chans = [resample_poly(x[:, c], up, down).astype(np.float32) for c in range(x.shape[1])]
    m = min(len(ch) for ch in chans)
    return np.stack([ch[:m] for ch in chans], axis=1)

def match_ir_channels_to_output(ir: np.ndarray, out_channels: int) -> np.ndarray:
    c_ir = ir.shape[1]
    if c_ir == out_channels:
        return ir
    if c_ir == 1 and out_channels > 1:
        return np.repeat(ir, out_channels, axis=1)
    if c_ir > 1 and out_channels == 1:
        # mixdown to mono
        return np.mean(ir, axis=1, keepdims=True).astype(np.float32)
    # default: crop or tile
    if c_ir > out_channels:
        return ir[:, :out_channels]
    return np.repeat(ir, out_channels, axis=1)

# ---------- FDL Convolver (mono in -> C_out out) ----------
class FDLConvolver:
    def __init__(self, ir: np.ndarray, block: int):
        """
        ir: (M, C_out), block: L
        Mono input. Output channels = ir.shape[1].
        """
        self.L = int(block)
        self.Nfft = 2 * self.L
        self.F = self.Nfft // 2 + 1
        self.C_out = ir.shape[1]

        # Partition IR and precompute spectra per output channel
        M = ir.shape[0]
        self.K = int(np.ceil(M / self.L))
        self.H = np.empty((self.C_out, self.K, self.F), dtype=np.complex64)
        for co in range(self.C_out):
            hpad = np.zeros(self.K * self.L, dtype=np.float32)
            hpad[:M] = ir[:, co]
            for k in range(self.K):
                seg = hpad[k*self.L:(k+1)*self.L]
                self.H[co, k, :] = rfft(np.pad(seg, (0, self.L)))  # 2L FFT

        # State
        self.Xring = np.zeros((self.K, self.F), dtype=np.complex64)
        self.ridx = 0
        self.overlap = np.zeros((self.C_out, self.L), dtype=np.float32)

    def process_block(self, x_block_mono: np.ndarray) -> np.ndarray:
        """
        x_block_mono: (L, 1) float32 in [-1,1]
        returns: (L, C_out) float32
        """
        x = x_block_mono[:, 0]
        X_i = rfft(np.pad(x, (0, self.L)))     # size 2L -> F bins

        self.Xring[self.ridx, :] = X_i
        # Accumulate per output channel
        Yc = np.zeros((self.C_out, self.F), dtype=np.complex64)
        for co in range(self.C_out):
            Y = np.zeros(self.F, dtype=np.complex64)
            ridx = self.ridx
            for k in range(self.K):
                Y += self.H[co, k, :] * self.Xring[ridx, :]
                ridx = (ridx - 1) % self.K
            Yc[co, :] = Y

        # IFFT + OLA
        y = np.empty((self.C_out, self.L), dtype=np.float32)
        for co in range(self.C_out):
            y_time = irfft(Yc[co, :]).astype(np.float32)   # length 2L
            y_blk  = y_time[:self.L] + self.overlap[co]
            self.overlap[co] = y_time[self.L:]
            y[co, :] = y_blk

        self.ridx = (self.ridx + 1) % self.K
        return y.T  # (L, C_out)

def run_realtime_convolution(
    ir_path: str,
    block: int = BLOCK,
    in_device_index=None,
    out_device_index=None,
    sample_rate=None,
    wet: float = 1.0,
    dry: float = 0.0,
):
    # Load IR and make it match the stream config
    ir, fs_ir = load_ir_any(ir_path)

    # Temporarily open stream to get fs and output channels
    # We know input is mono, output channels == IR channels
    out_channels = ir.shape[1] if ir.ndim == 2 else 1

    # # Open stream now; it returns actual fs to use (if not forced)
    # p, stream, fs = open_duplex_stream(
    #     convolver_cb=None,  # placeholder; we attach later
    #     in_dev=in_device_index,
    #     out_dev=out_device_index,
    #     fs=sample_rate,
    #     out_channels=out_channels
    # )
    # # Now fs is known; resample IR if needed and fix channel count
    ir = resample_if_needed(ir, fs_ir, 48000)
    # ir = match_ir_channels_to_output(ir, out_channels)

    # Build DSP
    convolver = FDLConvolver(ir, block=block)

    # Define callback that uses the convolver
    def dsp_callback(x_block_mono):
        # wet/dry (dry duplicates across output channels)
        wet_block = convolver.process_block(x_block_mono)               # (L, C_out)
        if dry != 0.0:
            dry_block = np.repeat(x_block_mono, wet_block.shape[1], axis=1)
            y = dry * dry_block + wet * wet_block
        else:
            y = wet * wet_block
        return y

    p2, stream2, fs2 = open_duplex_stream(
        convolver_cb=dsp_callback,
        in_dev=in_device_index,
        out_dev=out_device_index,
        fs=48000,
        out_channels=out_channels
    )

    try:
        stream2.start_stream()
        print(f"[RT] running at fs={fs2} Hz, block={block}, in:1ch, out:{out_channels}ch. Press Ctrl+C to stop.")
        while stream2.is_active():
            pass
    except KeyboardInterrupt:
        print("\n[RT] stopping…")
    finally:
        stream2.stop_stream()
        stream2.close()
        p2.terminate()

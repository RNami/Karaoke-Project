# Filters/ir_utils.py
import os, math
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly
import soundfile as sf

# === Minimal IR loader & FDL convolver =======================================
COMMON_IR_KEYS = ["h_air", "ir", "h", "rir", "impulse_response", "IR", "H"]
COMMON_INFO_KEYS = ["air_info", "info", "AIR_info"]


def _extract_scalar(v):
    if isinstance(v, np.ndarray):
        v = np.squeeze(v)
        if v.size == 1:
            return float(v)
    try:
        return float(v)
    except Exception:
        return None


def _get_struct_field(struct_obj, name: str):
    if hasattr(struct_obj, "dtype") and struct_obj.dtype is not None and struct_obj.dtype.names:
        try:
            field = struct_obj[name]
        except Exception:
            return None
        return np.squeeze(field)
    return None


def _pick_ir_array(d: dict) -> np.ndarray:
    for k in COMMON_IR_KEYS:
        if k in d and isinstance(d[k], np.ndarray):
            return np.atleast_2d(d[k]).astype(np.float32)
    # fallback: largest numeric array
    cands = []
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim in (1, 2) and np.issubdtype(v.dtype, np.number):
            cands.append((k, v))
    if not cands:
        raise RuntimeError("No numeric IR array found in .mat file.")
    _, v = max(cands, key=lambda kv: kv[1].size)
    return np.atleast_2d(v).astype(np.float32)


def _find_fs_in_info_struct(d: dict):
    for info_key in COMMON_INFO_KEYS:
        if info_key in d:
            info = d[info_key]
            for cand in ["fs", "Fs", "FS", "sampling_rate", "sr", "sample_rate", "samplingRate"]:
                field = _get_struct_field(info, cand)
                if field is None:
                    continue
                val = _extract_scalar(field)
                if val and val > 0:
                    return float(val)
    return None


def load_ir_any(ir_path: str):
    """Load .mat or .wav IR. Returns (ir_array (M, C), fs)."""
    p = os.path.abspath(ir_path)
    if not os.path.exists(p):
        raise FileNotFoundError(ir_path)
    base, ext = os.path.splitext(p)
    ext = ext.lower()
    if ext == ".mat":
        d = loadmat(p, squeeze_me=True)
        ir = _pick_ir_array(d)
        # ensure shape (M, C)
        if ir.ndim == 1:
            ir = ir[:, None]
        if ir.ndim == 2 and ir.shape[0] < 8 and ir.shape[1] > 8:
            ir = ir.T
        fs = _find_fs_in_info_struct(d)
        if fs is None:
            for cand in ["fs", "Fs", "FS", "sampling_rate", "sr", "sample_rate", "samplingRate"]:
                if cand in d:
                    fs = _extract_scalar(d[cand])
                    if fs:
                        break
        if fs is None:
            raise RuntimeError("IR sample rate not found in .mat")
        return ir.astype(np.float32), int(round(fs))
    elif ext in (".wav", ".flac", ".aiff", ".aif"):
        ir, fs = sf.read(p, always_2d=True)
        return ir.astype(np.float32), int(round(fs))
    else:
        raise RuntimeError(f"Unsupported IR path: {ir_path}")


def resample_if_needed(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return x
    g = math.gcd(fs_out, fs_in)
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
        return np.mean(ir, axis=1, keepdims=True).astype(np.float32)
    if c_ir > out_channels:
        return ir[:, :out_channels]
    return np.repeat(ir, out_channels, axis=1)


# === Helper: multi-channel resample ==========================================
def resample_multichannel(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """x shape either (N,) or (N, C). Returns resampled same shape convention."""
    if fs_in == fs_out:
        return x
    if x.ndim == 1:
        return resample_poly(x, fs_out, fs_in)
    # per-channel
    chans = [resample_poly(x[:, c], fs_out, fs_in) for c in range(x.shape[1])]
    m = min(len(ch) for ch in chans)
    return np.stack([ch[:m] for ch in chans], axis=1)

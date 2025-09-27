# audio_engine.py
import os
import threading
import math

import pyaudio
import numpy as np
from scipy.signal import lfilter, resample_poly
from scipy.io import loadmat
import soundfile as sf
from numpy.fft import rfft, irfft
from collections import deque

FORMAT = pyaudio.paInt16
BUFFER_SIZE = 1024   # fixed internal buffer, GUI can still show value if desired

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

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
        

        self.pitch_history = deque(maxlen=5)   # smooth frequency
        self.note_history = deque(maxlen=5)    # smooth note decisions

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

        self.last_X = X_i  # store for pitch detection
        self.ridx = (self.ridx + 1) % self.K
        return y.T  # (L, C_out)

    def detect_pitch(self, fs: int) -> tuple[float, str]:
        """
        Estimate pitch from the most recent FFT block (self.last_X).
        Smooths both frequency and note decisions.
        Returns (frequency Hz, note name).
        """
        if not hasattr(self, "last_X"):
            return 0.0, "N/A"

        spectrum = np.abs(self.last_X)
        spectrum[0] = 0  # remove DC

        # --- Ignore very low frequencies (below ~50 Hz) ---
        min_freq = 50.0
        min_bin = int(min_freq * self.Nfft / fs)
        spectrum[:min_bin] = 0

        # --- Find peak ---
        peak_idx = np.argmax(spectrum)
        if spectrum[peak_idx] < 1e-6:  # silence / noise floor
            return 0.0, "N/A"

        # --- Parabolic interpolation for sub-bin accuracy ---
        if 1 <= peak_idx < len(spectrum) - 1:
            alpha = spectrum[peak_idx - 1]
            beta = spectrum[peak_idx]
            gamma = spectrum[peak_idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak_idx = peak_idx + p

        freq = peak_idx * fs / self.Nfft

        # --- Smooth frequency ---
        self.pitch_history.append(freq)
        freq_smoothed = sum(self.pitch_history) / len(self.pitch_history)

        if freq_smoothed <= 0:
            return 0.0, "N/A"

        # --- Convert frequency -> MIDI note ---
        midi_num = int(round(69 + 12 * np.log2(freq_smoothed / 440.0)))
        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        note_name = note_names[midi_num % 12]
        octave = midi_num // 12 - 1
        note_str = f"{note_name}{octave}"

        # --- Smooth note decisions ---
        self.note_history.append(note_str)
        note_smoothed = max(set(self.note_history), key=self.note_history.count)

        return freq_smoothed, note_smoothed






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
        self.current_note = "C"
        self.lock = threading.Lock()

        # convolver related
        self.convolver = None
        self.ir_path = None
        self.ir_fs = None
        self.ir_channels = None
        self.wet = 1.0
        self.dry = 0.0

    def load_ir(self, ir_path: str, target_fs: int):
        """
        Load IR from .mat/.wav and resample it to target_fs (input stream fs).
        This also sets up the FDLConvolver with block size BUFFER_SIZE.
        """
        ir, fs_ir = load_ir_any(ir_path)
        # resample to target_fs
        if fs_ir != target_fs:
            ir = resample_if_needed(ir, fs_ir, target_fs)
            fs_ir = target_fs
        self.ir_path = ir_path
        self.ir_fs = fs_ir
        self.ir_channels = ir.shape[1]
        # convolver expects (M, C_out)
        self.convolver = FDLConvolver(ir, block=BUFFER_SIZE)

    def set_wet_dry(self, wet: float, dry: float):
        self.wet = float(wet)
        self.dry = float(dry)

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

        # If the user picked Convolver but no IR loaded -> try to load default (if path set)
        if self.effect_name == "Convolver" and self.convolver is None and self.ir_path is not None:
            try:
                self.load_ir(self.ir_path, target_fs=self.in_rate)
            except Exception as e:
                print(f"[AudioEngine] failed to load IR at start: {e}")
                self.convolver = None

        self.running = True

        # Start worker thread
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

            # convert to numpy int16 and reshape into channels
            x = np.frombuffer(data, dtype=np.int16)
            if self.in_channels > 1:
                try:
                    x = x.reshape(-1, self.in_channels)
                except Exception:
                    # fallback: if length not divisible, trim
                    n_frames = len(x) // self.in_channels
                    x = x[: n_frames * self.in_channels].reshape(n_frames, self.in_channels)

            # RMS level (mono)
            x_mono_for_level = x.mean(axis=1) if x.ndim > 1 else x
            self.current_level = min(100.0, (np.sqrt(np.mean(x_mono_for_level.astype(np.float32) ** 2)) / 32768.0) * 100.0)

            # Detect Pitch
            if self.effect_name == "Convolver" and self.convolver is not None:
                self.current_note = self.convolver.detect_pitch(fs=self.ir_fs)
            else:
                self.current_note = "asdfhusadfsfihdsafdsafdsa/A"

            # Apply effect
            if self.effect_name == "Robot Voice":
                # operate on int16 1D stream; convert if needed
                if x.ndim > 1:
                    x_proc = x.mean(axis=1).astype(np.int16)
                else:
                    x_proc = x
                x = robot_voice_effect(x_proc, self.in_rate)

            elif self.effect_name == "Concert Hall":
                if x.ndim > 1:
                    x_proc = x.mean(axis=1).astype(np.int16)
                else:
                    x_proc = x
                x = concert_hall_effect(x_proc, self.in_rate)

            elif self.effect_name == "Convolver" and self.convolver is not None:
                # Convert input to float32 in [-1,1] and mono as required by convolver
                if x.ndim > 1:
                    x_mono = x.mean(axis=1).astype(np.float32) / 32768.0
                else:
                    x_mono = x.astype(np.float32) / 32768.0
                # ensure block length matches BUFFER_SIZE
                if len(x_mono) != BUFFER_SIZE:
                    # pad or truncate
                    if len(x_mono) < BUFFER_SIZE:
                        x_mono = np.pad(x_mono, (0, BUFFER_SIZE - len(x_mono)))
                    else:
                        x_mono = x_mono[:BUFFER_SIZE]
                x_block = x_mono[:, None]  # (L,1)
                try:
                    y_block = self.convolver.process_block(x_block)  # (L, C_out) float32
                except Exception as e:
                    print(f"[AudioEngine] convolver process error: {e}")
                    y_block = np.repeat(x_block, self.out_channels, axis=1)

                # wet/dry mix
                if self.dry != 0.0:
                    dry_block = np.repeat(x_block, y_block.shape[1], axis=1)
                    y_mixed = self.dry * dry_block + self.wet * y_block
                else:
                    y_mixed = self.wet * y_block

                # match to output channels
                if y_mixed.shape[1] < self.out_channels:
                    y_mixed = np.repeat(y_mixed, math.ceil(self.out_channels / y_mixed.shape[1]), axis=1)[:, :self.out_channels]
                elif y_mixed.shape[1] > self.out_channels:
                    y_mixed = y_mixed[:, :self.out_channels]

                # convert back to int16 interleaved
                x = np.clip((y_mixed * 32767.0), -32768, 32767).astype(np.int16)

            # If effect hasn't replaced x and we still have multi-channel int16 array, keep it
            # Resample if needed (handle multi-channel)
            if self.in_rate != self.out_rate:
                # resample expects float or int arrays; convert so resample_multichannel works
                if x.ndim == 1:
                    x = resample_poly(x.astype(np.float32), self.out_rate, self.in_rate).astype(np.int16)
                else:
                    # per-channel resample
                    chans = []
                    for c in range(x.shape[1]):
                        ch = resample_poly(x[:, c].astype(np.float32), self.out_rate, self.in_rate)
                        chans.append(ch)
                    m = min(len(ch) for ch in chans)
                    x = np.stack([ch[:m] for ch in chans], axis=1).astype(np.int16)

            # Channel match / shape adjustments before write
            # Ensure x is interleaved 1D bytes-ready or shape (N, out_channels)
            if self.out_channels > 1:
                if x.ndim == 1:
                    # duplicate mono
                    x = np.repeat(x[:, None], self.out_channels, axis=1)
                elif x.shape[1] != self.out_channels:
                    if x.shape[1] < self.out_channels:
                        x = np.repeat(x, math.ceil(self.out_channels / x.shape[1]), axis=1)[:, :self.out_channels]
                    else:
                        x = x[:, :self.out_channels]
            else:
                # out_channels == 1
                if x.ndim > 1:
                    x = x.mean(axis=1).astype(np.int16)

            # write safely
            with self.lock:
                if self.running and self.out_stream is not None:
                    try:
                        # out_stream.write expects bytes
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

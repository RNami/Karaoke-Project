# audio_engine.py
import os
import threading
import math

import pyaudio
import numpy as np
from scipy.signal import resample_poly
from scipy.io import loadmat
import soundfile as sf

from Filters.filters import FDLConvolver, AudioEffects
from Filters.ir_utils import load_ir_any, resample_if_needed

FORMAT = pyaudio.paInt16
BUFFER_SIZE = 256   # fixed internal buffer, GUI can still show value if desired

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

            # Apply effect
            if self.effect_name == "Robot Voice":
                # operate on int16 1D stream; convert if needed
                if x.ndim > 1:
                    x_proc = x.mean(axis=1).astype(np.int16)
                else:
                    x_proc = x
                x = AudioEffects.robot_voice_effect(x_proc, self.in_rate)

            elif self.effect_name == "Concert Hall":
                if x.ndim > 1:
                    x_proc = x.mean(axis=1).astype(np.int16)
                else:
                    x_proc = x
                x = AudioEffects.concert_hall_effect(x_proc, self.in_rate)

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

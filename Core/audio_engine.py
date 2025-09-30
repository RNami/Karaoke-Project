# Core/audio_engine.py
import math
import threading

import numpy as np
import pyaudio
from scipy.signal import resample_poly


from Filters.filter_improved import FDLConvolver, concert_hall_effect, robot_voice_effect
from Filters.ir_utils import load_ir_any, resample_if_needed
from note_detection.note_detection import NoteDetection


FORMAT = pyaudio.paInt16
BUFFER_SIZE = 128  # fixed internal buffer, GUI can still show value if desired


# === Engine ===================================================================
class AudioEngine:
    """
    Real-time audio engine that manages microphone input, speaker output,
    and effect processing.

    The engine opens two independent PyAudio streams (input + output),
    and runs an I/O loop in a worker thread:
    - Reads blocks of audio from the microphone.
    - Optionally applies an effect (Robot Voice, Concert Hall, or Convolver).
    - Resamples audio if input and output sample rates differ.
    - Writes processed audio to the output device.

    Attributes
    ----------
    pa : pyaudio.PyAudio
        Main PyAudio instance managing devices and streams.
    in_stream : pyaudio.Stream or None
        Active input stream (microphone).
    out_stream : pyaudio.Stream or None
        Active output stream (speakers).
    running : bool
        Whether the audio engine is currently active.
    effect_name : str
        Currently selected effect ("None", "Robot Voice", "Concert Hall", "Convolver").
    current_level : float
        RMS microphone level in range [0, 100].
    current_note : str
        Last detected pitch class (e.g., "C#", "F").
    convolver : FDLConvolver or None
        Convolution engine for IR processing, if loaded.
    ir_path : str or None
        Path to the currently loaded impulse response (if any).
    ir_fs : int or None
        Sample rate of the loaded impulse response.
    ir_channels : int or None
        Number of channels in the impulse response.
    wet : float
        Scaling factor for processed (wet) signal.
    dry : float
        Scaling factor for unprocessed (dry) signal.
    note_detector : NoteDetection or None
        Pitch detection helper, optionally used during processing.
    lock : threading.Lock
        Synchronization for thread-safe stream operations.
    """

    def __init__(self):
        """Initialize the audio engine and prepare internal state."""
        self.pa = pyaudio.PyAudio()
        self.in_stream = None
        self.out_stream = None
        self.running = False
        self.effect_name = "None"
        self.current_level = 0.0
        self.current_note = "C"
        self.lock = threading.Lock()

        # Effect-related state
        self.convolver = None
        self.ir_path = None
        self.ir_fs = None
        self.ir_channels = None
        self.wet = 1.0
        self.dry = 0.0

        self.note_detector = NoteDetection(block=BUFFER_SIZE)
        self.rir_measuring = False
        self.rir_buffer = []
        self.rir_recorder = None

    def load_ir(self, ir_path: str, target_fs: int):
        """
        Load an impulse response (IR) and configure the convolver.

        Parameters
        ----------
        ir_path : str
            Path to the impulse response file (.mat, .wav, etc.).
        target_fs : int
            Target sampling rate for resampling the IR.

        Raises
        ------
        ValueError
            If the IR cannot be loaded or resampled.
        """
        ir, fs_ir = load_ir_any(ir_path)

        # Resample if needed
        if fs_ir != target_fs:
            ir = resample_if_needed(ir, fs_ir, target_fs)
            fs_ir = target_fs

        self.ir_path = ir_path
        self.ir_fs = fs_ir
        self.ir_channels = ir.shape[1]

        #  Setup FDLConvolver with fixed block size
        self.convolver = FDLConvolver(ir, block=BUFFER_SIZE)

        # Initialize pitch detection with same block size
        self.note_detector = NoteDetection(block=BUFFER_SIZE)

    def set_wet_dry(self, wet: float, dry: float):
        """
        Adjust wet/dry mixing levels for the convolver effect.

        Parameters
        ----------
        wet : float
            Proportion of the processed (wet) signal [0–1].
        dry : float
            Proportion of the unprocessed (dry) signal [0–1].
        """
        self.wet = float(wet)
        self.dry = float(dry)

    def start_stream(self, inp_idx, out_idx, effect_name="None"):
        """
        Start the input/output audio streams and worker thread.

        Parameters
        ----------
        inp_idx : int
            Index of the input (microphone) device.
        out_idx : int
            Index of the output (speaker) device.
        effect_name : str, optional
            Name of the effect to apply. Default is "None".

        Notes
        -----
        This method opens two independent PyAudio streams:
        - Input-only stream (microphone).
        - Output-only stream (speakers).
        """
        # Query device defaults
        in_dev = self.pa.get_device_info_by_index(inp_idx)
        out_dev = self.pa.get_device_info_by_index(out_idx)
        self.in_rate = int(in_dev["defaultSampleRate"])
        self.out_rate = int(out_dev["defaultSampleRate"])
        self.in_channels = max(1, min(in_dev["maxInputChannels"], 2))
        self.out_channels = max(1, min(out_dev["maxOutputChannels"], 2))
        self.effect_name = effect_name
        self.output_device_index = out_idx
        self.rir_measuring = False
        self.rir_buffer = []

        # Open input stream
        self.in_stream = self.pa.open(format=FORMAT,
                                      channels=self.in_channels,
                                      rate=self.in_rate,
                                      input=True,
                                      frames_per_buffer=BUFFER_SIZE,
                                      input_device_index=inp_idx)

        # Open output stream
        self.out_stream = self.pa.open(format=FORMAT,
                                       channels=self.out_channels,
                                       rate=self.out_rate,
                                       output=True,
                                       frames_per_buffer=BUFFER_SIZE,
                                       output_device_index=out_idx)

        # If convolver effect selected but not initialized, try to load IR
        if self.effect_name == "Convolver" and self.convolver is None and self.ir_path is not None:
            try:
                self.load_ir(self.ir_path, target_fs=self.in_rate)
            except Exception as e:
                print(f"[AudioEngine] failed to load IR at start: {e}")
                self.convolver = None

        # Launch processing thread
        self.running = True
        self.thread = threading.Thread(target=self._io_loop, daemon=True)
        self.thread.start()

    # -------------------------------------------------------------------------
    def _io_loop(self):
        """
        Worker loop for real-time audio I/O.

        This loop:
        - Reads a block of audio from the input stream.
        - Computes RMS level and updates `current_level`.
        - Applies pitch detection if enabled.
        - Applies the selected audio effect.
        - Resamples to match output rate.
        - Writes the processed audio to the output stream.
        """
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

            # capture for RIR if measuring
            if self.rir_measuring:
                # Convert to mono float32
                x_mono = x.mean(axis=1).astype(np.float32) / 32768.0 if x.ndim > 1 else x.astype(np.float32) / 32768.0
                self.rir_buffer.append(x_mono.copy())
            # RMS level (mono)
            x_mono_for_level = x.mean(axis=1) if x.ndim > 1 else x
            self.current_level = min(100.0, (np.sqrt(np.mean(x_mono_for_level.astype(np.float32) ** 2)) / 32768.0) * 100.0)

            # Note Detection (always runs, independent of effect)
            if self.note_detector is not None:
                freq, note = self.note_detector.process_block(x_mono_for_level.astype(np.int16), fs=self.in_rate)
                self.current_note = note

            # Apply effect
            if self.effect_name == "Robot Voice":
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
                if x.ndim > 1:
                    x_mono = x.mean(axis=1).astype(np.float32) / 32768.0
                else:
                    x_mono = x.astype(np.float32) / 32768.0
                if len(x_mono) != BUFFER_SIZE:
                    if len(x_mono) < BUFFER_SIZE:
                        x_mono = np.pad(x_mono, (0, BUFFER_SIZE - len(x_mono)))
                    else:
                        x_mono = x_mono[:BUFFER_SIZE]
                x_block = x_mono[:, None]  # (L,1)
                try:
                    y_block = self.convolver.process_block(x_block, note_detector=self.note_detector)
                except Exception as e:
                    print(f"[AudioEngine] convolver process error: {e}")
                    y_block = np.repeat(x_block, self.out_channels, axis=1)

                if self.dry != 0.0:
                    dry_block = np.repeat(x_block, y_block.shape[1], axis=1)
                    y_mixed = self.dry * dry_block + self.wet * y_block
                else:
                    y_mixed = self.wet * y_block

                if y_mixed.shape[1] < self.out_channels:
                    y_mixed = np.repeat(y_mixed, math.ceil(self.out_channels / y_mixed.shape[1]), axis=1)[:, :self.out_channels]
                elif y_mixed.shape[1] > self.out_channels:
                    y_mixed = y_mixed[:, :self.out_channels]

                x = np.clip((y_mixed * 32767.0), -32768, 32767).astype(np.int16)

            # Resample if needed
            if self.in_rate != self.out_rate:
                if x.ndim == 1:
                    x = resample_poly(x.astype(np.float32), self.out_rate, self.in_rate).astype(np.int16)
                else:
                    chans = []
                    for c in range(x.shape[1]):
                        ch = resample_poly(x[:, c].astype(np.float32), self.out_rate, self.in_rate)
                        chans.append(ch)
                    m = min(len(ch) for ch in chans)
                    x = np.stack([ch[:m] for ch in chans], axis=1).astype(np.int16)

            # Channel match before write
            if self.out_channels > 1:
                if x.ndim == 1:
                    x = np.repeat(x[:, None], self.out_channels, axis=1)
                elif x.shape[1] != self.out_channels:
                    if x.shape[1] < self.out_channels:
                        x = np.repeat(x, math.ceil(self.out_channels / x.shape[1]), axis=1)[:, :self.out_channels]
                    else:
                        x = x[:, :self.out_channels]
            else:
                if x.ndim > 1:
                    x = x.mean(axis=1).astype(np.int16)

            # Write out
            with self.lock:
                if self.running and self.out_stream is not None:
                    try:
                        self.out_stream.write(x.astype(np.int16).tobytes())
                    except OSError:
                        break

    # -------------------------------------------------------------------------
    def stop_stream(self):
        """
        Stop both input and output streams and end processing thread.

        This method is idempotent: calling it multiple times is safe.
        """
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

        # Wait for worker thread to terminate
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.5)

    # -------------------------------------------------------------------------
    def terminate(self):
        """
        Fully release resources and terminate the PyAudio instance.

        Should be called on application exit to avoid dangling device handles.
        """
        self.stop_stream()
        self.pa.terminate()

    def start_rir_recording(self):
        """
        Start buffering audio data for RIR measurement.
        """
        self.rir_buffer = []
        self.rir_measuring = True

    def stop_rir_recording(self):
        """
        Stop RIR measurement and return the recorded audio buffer.
        """
        self.rir_measuring = False

    @property
    def stream(self):
        """
        Return the active input stream.

        Returns
        -------
        pyaudio.Stream or None
            Active input stream if available, otherwise None.
        """
        return self.in_stream

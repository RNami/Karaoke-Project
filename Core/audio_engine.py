# Core/audio_engine.py
import numpy as np
import pyaudio
from Filters.filter_improved import FDLConvolver, robot_voice_effect, concert_hall_effect, BypassFilter
from Filters.ir_utils import load_ir_any as load_ir_file

class AudioEngine:
    """
    Real-time audio engine:
    - Captures input from microphone
    - Applies optional effects/filters
    - Plays output in real time

    Notes:
      - Input/Output PCM is handled as int16 in pyaudio callback.
      - Internal per-effect processing is done on normalized float32 in [-1, 1].
      - Convolver expects input blocks of length L == buffer_size; FDLConvolver is created when an IR is loaded.
    """

    def __init__(self, in_channels=1, out_channels=2, rate=48000, buffer_size=1024):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate = rate
        self.buffer_size = buffer_size

        self.input_device_index = None
        self.output_device_index = None

        # PyAudio
        self.pa = pyaudio.PyAudio()
        self._running = False
        self.stream = None

        # RIR recording (standalone)
        self.rir_buffer = []
        self._recording_rir = False

        # Audio processing
        self.current_filter = None
        self.effect_mode = "Bypass"
        self.effect_name = "None"
        self.filters = [] # Custom filter objects if any
        self.ir_path = None 
        self.ir_data = None # numpy array
        self.ir_convolver = None # Instance of FDLConvolver when IR loaded
        self.wet = 1.0
        self.dry = 0.0

        # GUI / monitoring
        self.current_level = 0
        self.current_note = ''

        # logging callback (set by GUI)
        self.log_callback = None


    def set_log_callback(self, callback):
        """Set a logging callback for GUI log output."""
        self.log_callback = callback

    def _log(self, msg: str):
        """Internal helper to send log messages."""
        if hasattr(self, "log_callback") and self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                # avoid raising errors from logger (GUI Thread)
                print("[AudioEngine] log callback failed.")
                print(msg)
        else:
            print(msg)

    # ----------------------------
    # Effect management
    # ----------------------------
    def set_effect(self, effect_name: str):
        self.effect_name = effect_name
        self._log(f"[AudioEngine] Effect changed -> {effect_name}")

    # ----------------------------
    # IR / filter / wet-dry management
    # ----------------------------
    # def load_ir(self, path: str):
    #     self.ir_path = path
    #     self.ir_data = load_ir_file(path)
    def load_ir(self, path):
        import scipy.io
        import numpy as np

        try:
            if path.endswith(".mat"):
                mat = scipy.io.loadmat(path)

                # Try to automatically find the IR variable
                keys = [k for k in mat.keys() if not k.startswith("__")]
                if not keys:
                    raise ValueError("No valid variables found in MAT file.")

                ir = mat[keys[0]]
                ir = np.array(ir, dtype=float)

                # If IR is multi-dimensional, reduce it to 2D (samples × channels)
                if ir.ndim > 2:
                    ir = np.squeeze(ir)

                if ir.dtype == object:
                    ir = np.stack([np.squeeze(x) for x in ir], axis=-1)

            else:
                # Handle .wav or .npy IRs
                if path.endswith(".wav"):
                    from scipy.io import wavfile
                    _, ir = wavfile.read(path)
                    ir = ir.astype(float)
                elif path.endswith(".npy"):
                    ir = np.load(path)
                else:
                    raise ValueError("Unsupported IR format")

            # Normalize and assign to self.ir as float32 in [-1,1]
            max_abs = np.max(np.abs(ir)) if ir.size else 1.0
            if max_abs == 0:
                raise ValueError("IR is all zeros")
            ir = ir.astype(np.float32) / float(max_abs)

            # Ensure 2D shape (M, C_out)
            if ir.ndim == 1:
                ir = ir[:, None]

            # If IR channels != output channels, expand or trim
            if ir.shape[1] != self.out_channels:
                # If IR is mono but output is stereo, duplicate columns
                if ir.shape[1] == 1 and self.out_channels > 1:
                    ir = np.tile(ir, (1, self.out_channels))
                else:
                    # If more channels than hardware, trim to out_channels
                    ir = ir[:, :self.out_channels]

            self.ir_data = ir  # floating point IR in [-1,1], shape (M, C_out)
            self.ir_path = path
            self._log(f"[AudioEngine] IR loaded: {path} (shape={ir.shape})")

            # Build FDLConvolver using engine buffer_size
            try:
                self.ir_convolver = FDLConvolver(self.ir_data, block=self.buffer_size)
                self._log(f"[AudioEngine] Created FDLConvolver (block={self.buffer_size}).")
            except Exception as e:
                self.ir_convolver = None
                self._log(f"[AudioEngine] Failed to build convolver: {e}")
                raise

        except Exception as e:
            self._log(f"Could not load IR:\n{e}")
            # re-raise if you prefer caller to handle it:
            # raise

    def set_filters(self, filters_list: list):
        self.filters = filters_list

    def set_wet_dry(self, wet: float, dry: float):
        self.wet = float(wet)
        self.dry = float(dry)

    # ----------------------------
    # Stream control
    # ----------------------------
    def start_stream(self, input_device_index=None, output_device_index=None, effect_name="None"):
        """Start real-time audio streaming with given devices and effect."""
        if self.stream:
            self._log("[AudioEngine] Stream already active — restarting...")
            self.stop_stream()

        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.effect_name = effect_name

        # Log startup info
        self._log("[AudioEngine] Starting stream...")
        self._log(f"  Input Device  : {input_device_index}")
        self._log(f"  Output Device : {output_device_index}")
        self._log(f"  Sample Rate   : {self.rate} Hz")
        self._log(f"  Buffer Size   : {self.buffer_size}")
        self._log(f"  Effect        : {self.effect_name}")


        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=self.in_channels,
                rate=self.rate,
                input=True,
                output=True,
                frames_per_buffer=self.buffer_size,
                input_device_index=input_device_index,
                output_device_index=output_device_index,
                stream_callback=self._callback
            )
            self.stream.start_stream()
            self.running = True
            self._log("[AudioEngine] Stream started successfully.")
        except Exception as e:
            self._log(f"[AudioEngine] Error starting stream: {e}")
            raise

    def stop_stream(self):
        """Stop audio stream and release resources."""
        if not self.stream:
            self._log("[AudioEngine] No active stream to stop.")
            return
        self._log("[AudioEngine] Stopping stream...")
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self._log("[AudioEngine] Stream stopped.")
        except Exception as e:
            self._log(f"[AudioEngine] Error stopping stream: {e}")
        # reset running and monitoring state
        self.running = False
        # reset level and note
        self.current_level = 0.0
        self.current_note = ''
        
    def terminate(self):
        self.stop_stream()
        try:
            self.pa.terminate()
        except Exception:
            pass

    # ----------------------------
    # PyAudio callback
    # ----------------------------
    def _callback(self, in_data, frame_count, time_info, status):
        audio_in = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.current_level = np.sqrt(np.mean((audio_in / 32768.0) ** 2)) * 100
        audio_out = self._process(audio_in)  # already int16 clipped
        return (audio_out.tobytes(), pyaudio.paContinue)

    # ----------------------------
    # Processing
    # ----------------------------
    def _process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the selected effect or convolution using the loaded FDLConvolver.
        audio: int16 block (1D) or int-like array from callback
        returns: int16 block (same length x channels flattened if needed)
        """

        # Convert to float32 in [-1,1]
        audio_f = audio.astype(np.float32) / 32768.0

        # Ensure 2D shape (L, channels)
        if audio_f.ndim == 1:
            audio_f = audio_f[:, None]  # (L, 1)

        # Mix input to mono if needed
        mono_in = audio_f if audio_f.shape[1] == 1 else np.mean(audio_f, axis=1, keepdims=True)

        # Initialize output arrays
        processed_mono = None
        processed_block = None

        # ------------------------
        # Apply effect
        # ------------------------
        if self.effect_name == "Robot Voice":
            norm = mono_in[:, 0]
            robotized = np.sign(norm) * np.sqrt(np.abs(norm))
            processed_mono = robotized[:, None]
        elif self.effect_name == "Concert Hall":
            decay = np.exp(-np.linspace(0, 1, 256))
            processed_mono = np.convolve(mono_in[:, 0], decay, mode="same")[:, None]
        elif self.effect_name == "Convolver":
            if self.ir_convolver is not None:
                processed_block = self.ir_convolver.process_block(mono_in)
            else:
                processed_mono = mono_in
        else:
            # Bypass
            processed_mono = mono_in

        # ------------------------
        # Prepare output block
        # ------------------------
        if processed_block is not None:
            out_block = processed_block
        else:
            # Expand mono to match output channels only if necessary
            if processed_mono.shape[1] == self.out_channels:
                out_block = processed_mono
            else:
                out_block = np.tile(processed_mono, (1, self.out_channels))

        # Prepare dry block once
        if audio_f.shape[1] == self.out_channels:
            dry_block = audio_f
        else:
            dry_block = np.tile(mono_in, (1, self.out_channels))

        # Wet/Dry mix
        out = (self.dry * dry_block) + (self.wet * out_block)

        # ------------------------
        # Safe clipping (Fix 2)
        out = np.clip(out, -1.0, 1.0)

        # Convert to int16
        out_int16 = (out * 32767.0).astype(np.int16)

        # Interleave channels properly (Fix 1)
        if out_int16.ndim == 2 and out_int16.shape[1] > 1:
            return out_int16.flatten('F')
        else:
            return out_int16.ravel()

    # ----------------------------
    # RIR Recording (standalone)
    # ----------------------------
    def start_rir_recording(self, filename: str = "rir_recording.wav"):
        """
        Starts a standalone RIR recording stream (no playback, no restart).
        """
        import wave

        if self._recording_rir:
            self._log("[AudioEngine] RIR recording already in progress.")
            return

        self._log(f"[AudioEngine] Starting RIR recording -> {filename}")
        self.rir_buffer = []
        self._recording_rir = True

        self._rir_wavefile = wave.open(filename, 'wb')
        self._rir_wavefile.setnchannels(self.in_channels)
        self._rir_wavefile.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
        self._rir_wavefile.setframerate(self.rate)

        # Create input-only stream
        self.rir_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.in_channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            input_device_index=self.input_device_index,
            stream_callback=self._rir_callback
        )
        self.rir_stream.start_stream()

    def stop_rir_recording(self):
        """Stop recording and close resources."""
        if not self._recording_rir:
            return

        self._log("[AudioEngine] Stopping RIR recording.")
        self._recording_rir = False

        if hasattr(self, "rir_stream") and self.rir_stream is not None:
            self.rir_stream.stop_stream()
            self.rir_stream.close()
            self.rir_stream = None

        if hasattr(self, "_rir_wavefile"):
            self._rir_wavefile.close()
            del self._rir_wavefile

    def _rir_callback(self, in_data, frame_count, time_info, status):
        if not self._recording_rir:
            return (None, pyaudio.paComplete)

        audio_block = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.rir_buffer.append(audio_block)

        if hasattr(self, "_rir_wavefile"):
            self._rir_wavefile.writeframes(in_data)

        return (None, pyaudio.paContinue)

    # ----------------------------
    # Stream state
    # ----------------------------
    def is_stream_active(self):
        """Return True if the main stream is active and running."""
        return self.stream is not None and self.stream.is_active()

    @property
    def running(self):
        """Return True if the main stream is active and running."""
        return self._running
    
    @running.setter
    def running(self, value: bool):
        self._running = value
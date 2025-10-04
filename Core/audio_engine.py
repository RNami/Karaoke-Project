# Core/audio_engine.py
import numpy as np
import pyaudio
from Filters.base_filters import FDLConvolver
from Filters.ir_utils import load_ir_any as load_ir_file

class AudioEngine:
    """
    Real-time audio engine:
    - Captures input from microphone
    - Applies optional effects/filters
    - Plays output in real time
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
        self.rir_buffer = []
        self._recording_rir = False

        # Audio processing
        self.effect_name = "None"
        self.filters = []
        self.ir_path = None
        self.ir_data = None
        self.wet = 1.0
        self.dry = 0.0

        # GUI / monitoring
        self.current_level = 0
        self.current_note = ''

    # ----------------------------
    # Stream control
    # ----------------------------
    def start_stream(self, input_device_index=None, output_device_index=None, effect_name="None"):
        if self.stream:
            self.stop_stream()

        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.effect_name = effect_name

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

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.running = False

    def terminate(self):
        self.stop_stream()
        self.pa.terminate()

    # ----------------------------
    # IR / filter / wet-dry management
    # ----------------------------
    def load_ir(self, path: str):
        self.ir_path = path
        self.ir_data = load_ir_file(path)

    def set_filters(self, filters_list: list):
        self.filters = filters_list

    def set_wet_dry(self, wet: float, dry: float):
        self.wet = wet
        self.dry = dry

    # ----------------------------
    # PyAudio callback
    # ----------------------------
    def _callback(self, in_data, frame_count, time_info, status):
        audio_in = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.current_level = np.sqrt(np.mean(audio_in**2))
        audio_out = self._process(audio_in)
        audio_out = np.clip(audio_out, -32768, 32767).astype(np.int16)
        return (audio_out.tobytes(), pyaudio.paContinue)

    # ----------------------------
    # Processing
    # ----------------------------
    def _process(self, audio: np.ndarray) -> np.ndarray:
        processed = np.copy(audio)

        # Built-in effects
        if self.effect_name == "Robot Voice":
            processed = self._robot_effect(processed)
        elif self.effect_name == "Concert Hall":
            processed = self._reverb_effect(processed)
        elif self.effect_name == "Convolver" and self.ir_data is not None:
            processed = self._convolve_ir(processed)

        # Custom filters
        for f in self.filters:
            processed = FDLConvolver(processed, f)

        return self.dry * audio + self.wet * processed

    # ----------------------------
    # Effect implementations
    # ----------------------------
    def _robot_effect(self, audio: np.ndarray) -> np.ndarray:
        return np.sign(audio) * np.sqrt(np.abs(audio))

    def _reverb_effect(self, audio: np.ndarray) -> np.ndarray:
        decay = np.exp(-np.linspace(0, 1, 256))
        return np.convolve(audio, decay, mode='same')

    def _convolve_ir(self, audio: np.ndarray) -> np.ndarray:
        return np.convolve(audio, self.ir_data, mode='same') if self.ir_data is not None else audio

    # ----------------------------
    # RIR Recording (standalone)
    # ----------------------------
    def start_rir_recording(self, filename: str = "rir_recording.wav"):
        """
        Starts a standalone RIR recording stream (no playback, no restart).
        """
        import wave

        if self._recording_rir:
            print("[AudioEngine] RIR recording already in progress.")
            return

        print(f"[AudioEngine] Starting RIR recording -> {filename}")
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

        print("[AudioEngine] Stopping RIR recording.")
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
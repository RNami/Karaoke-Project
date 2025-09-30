# measure_rir.py
import time
from threading import Thread

import numpy as np
import sounddevice as sd

class RIRRecorder:
    CHUNK = 1024  # default, will be overwritten by buffer_size

    def __init__(self, in_stream, in_rate, in_channels, output_device_index,
                 sweep_file: str, record_file: str,
                 record_length: int = 20):
        """
        RIR measurement class.

        Parameters
        ----------
        in_stream : pyaudio.Stream
            Active input stream from AudioEngine.
        in_rate : int
            Sample rate of input stream.
        in_channels : int
            Number of channels of input stream.
        output_device_index : int
            Output device index for sweep playback.
        sweep_file : str
            Path to the sine sweep WAV file.
        record_file : str, optional
            File to save the recorded RIR. Default is 'measured_room_impulse.wav'.
        record_length : int, optional
            Recording duration in seconds. Default is 20.
        """
        self.in_stream = in_stream
        self.in_rate = in_rate
        self.in_channels = in_channels
        self.output_device_index = output_device_index
        self.sweep_file = sweep_file
        self.record_file = record_file
        self.record_length = record_length
        self.buffer = []

        # Dynamically match chunk size to input stream if possible
        self.CHUNK = in_stream._frames_per_buffer if hasattr(in_stream, "_frames_per_buffer") else 1024

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)
    
    def _play_sweep(self, sweep, fs):
        self._log("[RIRRecorder] Playing sweep...")
        sd.play(sweep, fs)
        sd.wait()
        self._log("[RIRRecorder] Sweep finished.")

    def _record_to_file(self):
        import wave
        self._log("[RIRRecorder] Recording started (capturing from engine)...")

        # Tell engine to start recording
        self.in_stream.start_rir_recording()  # <-- engine method
        time.sleep(self.record_length)        # wait for sweep + measurement
        self.in_stream.stop_rir_recording()

        # Concatenate captured blocks
        data = np.concatenate(self.in_stream.rir_buffer)
        # Convert back to int16
        data_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)

        # Write to WAV
        with wave.open(self.record_file, "wb") as wf:
            wf.setframerate(self.in_rate)
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)
            wf.writeframes(data_int16.tobytes())

        self._log(f"[RIRRecorder] Recording finished. Saved to {self.record_file}")


    def measure(self):
        import soundfile as sf
        sweep, fs = sf.read(self.sweep_file)

        playback_thread = Thread(target=self._play_sweep, args=(sweep, fs))
        record_thread = Thread(target=self._record_to_file)

        # Start recording first
        record_thread.start()
        time.sleep(5)  # ensure recording ready
        playback_thread.start()

        record_thread.join()
        playback_thread.join()

        self._log("[RIRRecorder] Measurement complete.")

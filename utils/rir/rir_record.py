# measure_rir.py
import time
from threading import Thread

import numpy as np
import sounddevice as sd

import utils.rir.rir_postprocessing as rirPP

class RIRRecorder:
    CHUNK = 1024  # default, will be overwritten by buffer_size

    def __init__(self,
                in_stream,
                in_rate,
                in_channels,
                output_device_index,
                sweep_file: str,
                record_file: str,
                current_blocksize:int=256,
                record_length: int = 20,
                log_callback=None,
                progress_callback=None,
    ):
        """
        RIR measurement class.

        Parameters
        ----------
        in_stream : AudioEngine
            Active input stream from AudioEngine.
        in_rate : int
            Sample rate of input stream.
        in_channels : int
            Number of channels of input stream.
        output_device_index : int
            Output device index for sweep playback.
        sweep_file : str
            Path to the sine sweep WAV file.
        record_file : str
            File to save the recorded RIR.
        record_length : int, optional
            Recording duration in seconds. Default is 20.
        log_callback : callable, optional
            Function for logging messages to GUI or console.
        progress_callback : callable, optional
            Function to report progress (0.0-1.0 float).
        """
        self.in_stream = in_stream
        self.in_rate = in_rate
        self.in_channels = in_channels
        self.output_device_index = output_device_index
        self.sweep_file = sweep_file
        self.record_file = record_file
        self.record_length = record_length
        self.buffer = []
        self.current_blocksize = current_blocksize

        self.log_callback = log_callback
        self.progress_callback = progress_callback

        # Dynamically match chunk size to input stream if possible
        self.CHUNK = (
            in_stream._frames_per_buffer
            if hasattr(in_stream, "_frames_per_buffer")
            else 1024
        )

    def _log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)
    
    def _progress(self, value):
        """Report progress as a float between 0 and 1."""
        if self.progress_callback:
            self.progress_callback(value)

    def _play_sweep(self, sweep, fs):
        self._log("[RIRRecorder] Playing sweep...")
        sd.play(sweep, fs)
        sd.wait()
        self._log("[RIRRecorder] Sweep finished.")

    def _record_to_file(self):
        import wave
        self._log("[RIRRecorder] Recording started (capturing from engine)...")
        self._progress(0.1)

        # Tell engine to start recording
        self.in_stream.start_rir_recording(filename=self.record_file)
        start_time = time.time()

        # Incremental progress during recording
        while time.time() - start_time < self.record_length:
            elapsed = time.time() - start_time
            self._progress(0.1 + 0.3 * (elapsed / self.record_length))
            time.sleep(0.25)

        self.in_stream.stop_rir_recording()
        self._log("[RIRRecorder] Recording stopped.")

        # Concatenate captured blocks
        if not hasattr(self.in_stream, "rir_buffer") or not self.in_stream.rir_buffer:
            self._log("[RIRRecorder] Warning: No RIR data captured — buffer is empty.")
            return
        
        self._progress(0.65)
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
        self._progress(0.8)

    def measure(self):
        import soundfile as sf
        sweep, fs = sf.read(self.sweep_file)

        total_duration = max(self.record_length, len(sweep) / fs)
        start_time = time.time()

        # Start recording thread
        record_thread = Thread(target=self._record_to_file)
        record_thread.start()
        time.sleep(1.0)  # small buffer to ensure recording is active

        # Start sweep playback thread
        playback_thread = Thread(target=self._play_sweep, args=(sweep, fs))
        playback_thread.start()

        # --- Smoothed progress update loop ---
        prev_progress = 0.0
        smoothing_factor = 0.1  # smaller = smoother (e.g., 0.05–0.2 range)

        while record_thread.is_alive() or playback_thread.is_alive():
            elapsed = time.time() - start_time
            target_progress = min(elapsed / total_duration, 1.0)

            # Apply exponential smoothing
            prev_progress = (
                smoothing_factor * target_progress
                + (1 - smoothing_factor) * prev_progress
            )

            # Send to GUI
            if self.progress_callback:
                self.progress_callback(prev_progress)

            time.sleep(0.05)  # update every 50 ms

        # Join threads before finishing
        record_thread.join()
        playback_thread.join()

        # Ensure final progress = 100%
        if self.progress_callback:
            self.progress_callback(1.0)

        # Logging
        self._log("[RIRRecorder] Measurement complete.")
        self._log("[RIRRecorder] Postprocessing starting.")
        # rirPP.run_postprocessing(self.record_file, self.current_blocksize)
        self._log("[RIRRecorder] EQ File saved!")

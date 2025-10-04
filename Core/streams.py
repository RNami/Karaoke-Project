# Core/streams.py
import pyaudio
import numpy as np
import threading

class AudioStreamManager:
    def __init__(self, engine, input_index, output_index, buffer_size=1024):
        """
        Manage audio input/output streams and processing effects.
        """
        self.engine = engine
        self.input_index = input_index
        self.output_index = output_index
        self.buffer_size = buffer_size

        self.in_stream = None
        self.out_stream = None
        self.running = False

        self.effect_name = "None"
        self.current_level = 0.0
        self.current_note = ""

        # PyAudio instance
        self.pa = engine.pa

    def start(self):
        self.running = True
        self.in_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.engine.in_channels,
            rate=self.engine.in_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            input_device_index=self.input_index,
        )
        self.out_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.engine.out_channels,
            rate=self.engine.in_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            output_device_index=self.output_index,
        )

        # Start monitoring thread if needed
        self.thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.thread.start()

    def process_audio_loop(self):
        while self.running:
            try:
                data = self.in_stream.read(self.buffer_size, exception_on_overflow=False)
                # Apply effect (placeholder, add your actual processing)
                out_data = self.apply_effect(data)
                self.out_stream.write(out_data)
                # Optional: update level, pitch etc.
            except Exception as e:
                print(f"[AudioStreamManager] Error: {e}")
                break

    def apply_effect(self, data):
        # TODO: implement effects like Robot Voice, Convolver, etc.
        return data

    def stop(self):
        self.running = False
        if self.in_stream:
            self.in_stream.stop_stream()
            self.in_stream.close()
            self.in_stream = None
        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()
            self.out_stream = None

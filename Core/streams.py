# Core/streams.py
import threading
import numpy as np
import pyaudio

class AudioStream:
    def __init__(self, engine, input_index, output_index, buffer_size=1024):
        self.engine = engine
        self.pa = engine.pa
        self.input_index = input_index
        self.output_index = output_index
        self.buffer_size = buffer_size
        self.running = False

        self.in_stream = None
        self.out_stream = None
        self.thread = None

    def start(self):
        self.running = True
        self.in_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.engine.in_channels,
            rate=self.engine.in_rate,
            input=True,
            input_device_index=self.input_index,
            frames_per_buffer=self.buffer_size
        )
        self.out_stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.engine.out_channels,
            rate=self.engine.in_rate,
            output=True,
            output_device_index=self.output_index,
            frames_per_buffer=self.buffer_size
        )
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.running:
            data = self.in_stream.read(self.buffer_size, exception_on_overflow=False)
            out_data = self.engine.process_audio_buffer(data)
            self.out_stream.write(out_data)
            self.engine.update_level_and_pitch(data)

    def stop(self):
        self.running = False
        if self.in_stream:
            self.in_stream.stop_stream()
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()

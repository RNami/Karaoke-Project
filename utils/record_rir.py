import wave
import sys
import time

import pyaudio

CHUNK = 1024
fs = 48000
RECORD_LENGTH = 10
BIT_FORMAT = pyaudio.paInt24

def record_to_file(input_device_index):
    with wave.open('measured_room_impulse.wav', 'wb') as writing_object:
        audio_port_api = pyaudio.PyAudio()
        writing_object.setframerate(fs)
        writing_object.setnchannels(1)
        writing_object.setsampwidth(audio_port_api.get_sample_size(BIT_FORMAT))

        audio_stream = audio_port_api.open(format=BIT_FORMAT, rate=fs, input=True, channels=1, input_device_index=input_device_index)

        print('RECORD IS RUNNING')

        # for each chunk with defined until we wrote all samples for the defined recording time 
        for _ in range (fs // CHUNK*RECORD_LENGTH):
            writing_object.writeframes(audio_stream.read(CHUNK))
        print('RECORD DONE!')

        audio_stream.close()
        audio_port_api.terminate()

if __name__ == '__main__':
    record_to_file(2)
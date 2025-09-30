import wave
import sys
import time
from threading import Thread

import pyaudio
import sounddevice as sd
import soundfile as sf

CHUNK = 1024
fs = 48000
RECORD_LENGTH = 20
BIT_FORMAT = pyaudio.paInt24

def play_sweep(sweep, fs):
    print('Play back') 
    sd.play(sweep, fs)
    sd.wait()
    print('Play back stopped')

def record_to_file(input_device_index):
    audio_port_api = pyaudio.PyAudio() 
    audio_stream = audio_port_api.open(format=BIT_FORMAT, 
                                       rate=fs, 
                                       input=True, 
                                       channels=1, 
                                       input_device_index=input_device_index)
    print(audio_port_api. get_device_info_by_index(input_device_index))

    with wave.open('measured_room_impulse.wav', 'wb') as writing_object:
        writing_object.setframerate(fs)
        writing_object.setnchannels(1)
        writing_object.setsampwidth(audio_port_api.get_sample_size(BIT_FORMAT))       
        # for each chunk with defined until we wrote all samples for the defined recording time 
        for _ in range (fs*RECORD_LENGTH // CHUNK):
            writing_object.writeframes(audio_stream.read(CHUNK))
    
    audio_stream.close()
    audio_port_api.terminate()


def measure_rir(input_device_index):
    sweep, fs = sf.read('Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav')
    play_back_thread = Thread(target=play_sweep, args=(sweep, fs))
    recording_thread = Thread(target= record_to_file, args=(input_device_index,))
    
    recording_thread.start()
    print('RECORD IS RUNNING')
    time.sleep(5)   # wait 5 seconds before playback

    print('Play back')
    play_back_thread.start()
    
    recording_thread.join()
    play_back_thread.join()
    print('RECORD DONE!')


if __name__ == '__main__':
    measure_rir(1)
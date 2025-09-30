import numpy as np
import soundfile as sf
from matplotlib import pyplot as plt
from scipy import fft
CHUNK = 256
START_SWEEP = 6


def import_impulse_response():
    x, fs = sf.read('IR Database/AIR_1_4/test.wav', always_2d=True)
    if x.shape[1] == 2:
        x = (x[:,0]+x[:,1])/2
    x /= np.max(x)
    return x, fs

def get_fft(x, fs, block_size=256):
    # Calculate FFT size needed to get num_bins in RFFT
    out_fft = block_size*2
    
    # Perform RFFT
    freq_bins = np.fft.rfft(x[:,0],out_fft,axis=0)
    return freq_bins

def design_inv_filter(X)


def plot_impulse(x,fs):
    start = 0
    num_x = x.shape[0]
    end = num_x/fs
    t = np.linspace(start, end, num_x)
    plt.plot(t,x)
    plt.show()

def plot_freq(X, fs):
    f = np.fft.fftfreq(len(X), 1/fs)
    print(X.shape)
    plt.semilogx(f,X)
    plt.show()


def cut_impulse(x, fs, window_duration=2, hop_time=0.5):
    quite_count = 0
    start_index = int(START_SWEEP*fs)
    noise = x[:start_index]
    rms_noise = np.sqrt(np.mean(noise**2))
    signal = x[start_index:]

    window_size = int(window_duration*fs) # in samples
    hop_size = int(hop_time*fs) # in samples
    num_frames = 1 + (len(signal) - window_size) // hop_size

    for n in range(num_frames):
        start = n*hop_size
        end = start + window_size
        sig_windowed = signal[start:end]
        rms = np.sqrt(np.mean(sig_windowed**2))
        if rms < rms_noise:
            quite_count+=1
            if quite_count > 2:
                return signal[:end]
    return signal

x, fs = import_impulse_response()
plot_impulse(x, fs)
x = cut_impulse(x,fs)
plot_impulse(x, fs)
X_fft = get_fft(x, fs)
plot_freq(X_fft, fs)
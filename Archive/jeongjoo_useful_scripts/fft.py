import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.io.wavfile as wav

# Load input wav
fs_wav, audio = wav.read("output.wav")
audio = audio.astype(np.float32)

N = len(audio)              # number of samples
T = 1.0 / fs_wav            # sampling interval

# FFT
yf = fft(audio)
xf = fftfreq(N, T)[:N//2]   # frequency bins (only positive half)

# Plot amplitude spectrum
plt.figure(figsize=(16, 8))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram

# Load input wav
fs_wav, audio = wav.read("output.wav")
audio = audio.astype(np.float32)

# If stereo, take one channel
if audio.ndim > 1:
    audio = audio[:, 0]

# Spectrogram
f, t, Sxx = spectrogram(audio, fs=fs_wav, nperseg=1024, noverlap=512)

# Convert to dB scale
Sxx_dB = 10 * np.log10(Sxx + 1e-10)

# Plot
plt.figure(figsize=(16, 8))
plt.pcolormesh(t, f, Sxx_dB, shading="gouraud", cmap="inferno")
plt.colorbar(label="Power (dB)")
plt.title("Spectrogram (Time-Frequency Analysis)")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.ylim(0, fs_wav / 2)  # Nyquist limit
plt.show()

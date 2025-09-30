import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
audio_path = "boehmischer.wav"
y_perc, sr = librosa.load(audio_path, sr=None)  # keep original sampling rate
y = librosa.effects.harmonic(y_perc, margin=3.0)

# Compute STFT
S = np.abs(librosa.stft(y, n_fft=8192, hop_length=1024))

# Compute chroma (12 pitch classes)
chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=12)

# Sum energy over time for each note
chroma_sum = np.sum(chroma, axis=1)

# Note names
note_names = ["C", "C#", "D", "D#", "E", "F", 
              "F#", "G", "G#", "A", "A#", "B"]

# Sort by energy
sorted_idx = np.argsort(chroma_sum)[::-1]
notes_sorted = [note_names[i] for i in sorted_idx]
energy_sorted = chroma_sum[sorted_idx]


print("Top 5 notes:", notes_sorted[:7])

# Plot
# plt.figure(figsize=(12, 6))
# plt.bar(notes_sorted, energy_sorted, color='skyblue')
# plt.title("Most Active Notes (Chromagram, Octaves Ignored)")
# plt.xlabel("Note")
# plt.ylabel("Summed Chroma Energy")
# plt.grid(axis="y")
# plt.show()

# Plot2
plt.figure(figsize=(16, 6))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, cmap='coolwarm')
plt.colorbar(label='Chroma Energy')
plt.title("Chroma Representation")
plt.show()

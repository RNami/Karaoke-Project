import numpy as np
from collections import deque

class NoteDetection:
    def __init__(self, ir: np.ndarray, block: int):
        self.L = int(block)
        self.Nfft = 2 * self.L
        self.F = self.Nfft // 2 + 1
        self.C_out = ir.shape[1]

        self.pitch_history = deque(maxlen=5)   # smooth frequency
        self.note_history = deque(maxlen=5)    # smooth note decisions

        self.last_X = None  # last FFT block for pitch detection
        

    def detect_pitch(self, fs: int) -> tuple[float, str]:
        """
        Estimate pitch from the most recent FFT block (self.last_X).
        Smooths both frequency and note decisions.
        Returns (frequency Hz, note name).
        """
        # if not hasattr(self, "last_X"):
        #     print('DEBUG: No FFT block available for pitch detection.')
        #     return 0.0, "N/A"

        if self.last_X is None:
            print('DEBUG: No FFT block available for pitch detection.')
            return 0.0, "N/A"

        spectrum = np.abs(self.last_X)
        spectrum[0] = 0  # remove DC

        # --- Ignore very low frequencies (below ~50 Hz) ---
        min_freq = 50.0
        min_bin = int(min_freq * self.Nfft / fs)
        spectrum[:min_bin] = 0

        # --- Find peak ---
        peak_idx = np.argmax(spectrum)
        if spectrum[peak_idx] < 1e-6:  # silence / noise floor
            print('DEBUG: No significant peak found in spectrum.')
            return 0.0, "N/A"

        # --- Parabolic interpolation for sub-bin accuracy ---
        if 1 <= peak_idx < len(spectrum) - 1:
            alpha = spectrum[peak_idx - 1]
            beta = spectrum[peak_idx]
            gamma = spectrum[peak_idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak_idx = peak_idx + p

        freq = peak_idx * fs / self.Nfft

        # --- Smooth frequency ---
        self.pitch_history.append(freq)
        freq_smoothed = sum(self.pitch_history) / len(self.pitch_history)

        if freq_smoothed <= 0:
            print('DEBUG: Smoothed frequency is non-positive.')
            return 0.0, "N/A"

        # --- Convert frequency -> MIDI note ---
        midi_num = int(round(69 + 12 * np.log2(freq_smoothed / 440.0)))
        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        note_name = note_names[midi_num % 12]
        octave = midi_num // 12 - 1
        note_str = f"{note_name}{octave}"

        # --- Smooth note decisions ---
        self.note_history.append(note_str)
        note_smoothed = max(set(self.note_history), key=self.note_history.count)

        return freq_smoothed, note_smoothed

    def update_spectrum(self, spectrum: np.ndarray):
        """
        Store spectrum for later pitch detection.
        """
        self.last_X = spectrum
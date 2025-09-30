# note_detection/note_detection.py
import numpy as np
from collections import deque
from numpy.fft import rfft


class NoteDetection:
    def __init__(self, block: int):
        """
        Parameters
        ----------
        block : int
            Block size (L) used for FFT-based pitch detection.
        """
        self.L = int(block)
        self.Nfft = 2 * self.L
        self.F = self.Nfft // 2 + 1

        self.pitch_history = deque(maxlen=50)   # smooth frequency
        self.note_history = deque(maxlen=50)    # smooth note decisions
        self.pitch_threshold = 1e-3

        self.last_X = None  # last FFT spectrum

    # ------------------------------------------------------------------
    def process_block(self, x_block: np.ndarray, fs: int):
        """
        Take a mono block of time-domain audio and update spectrum.

        Parameters
        ----------
        x_block : np.ndarray
            Input block, shape (L,) or (L,1) int16/float32.
        fs : int
            Sampling rate in Hz.

        Returns
        -------
        tuple[float, str]
            (frequency in Hz, note string like 'A4').
        """
        if x_block.ndim > 1:
            x_block = x_block[:, 0]

        # normalize if int16
        if x_block.dtype.kind in "iu":
            x_block = x_block.astype(np.float32) / 32768.0

        self.last_X = rfft(np.pad(x_block, (0, self.L)))
        return self.detect_pitch(fs)

    # ------------------------------------------------------------------
    def detect_pitch(self, fs: int) -> tuple[float, str]:
        """
        Estimate pitch from the most recent FFT block.
        """
        if self.last_X is None:
            return 0.0, "N/A"

        spectrum = np.abs(self.last_X)
        spectrum[0] = 0  # remove DC

        # --- Ignore very low frequencies (below ~50 Hz) ---
        min_freq = 50.0
        min_bin = int(min_freq * self.Nfft / fs)
        spectrum[:min_bin] = 0

        # --- Find peak ---
        peak_idx = np.argmax(spectrum)
        peak_amp = spectrum[peak_idx]

        # --- Silence / too weak signal ---
        if peak_amp < self.pitch_threshold:
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

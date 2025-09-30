import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import lfilter

# === Audio effects ============================================================
class AudioEffects:
    def robot_voice_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
        freq = 80  # Hz
        t = np.arange(len(buffer)) / rate
        square_wave = np.sign(np.sin(2 * np.pi * freq * t))
        effected = buffer * square_wave
        return effected.astype(np.int16)


    def concert_hall_effect(buffer: np.ndarray, rate: int) -> np.ndarray:
        delay_ms = 300
        decay = 0.1
        delay_samples = int(rate * delay_ms / 1000)
        b = np.zeros(delay_samples + 1)
        b[0] = 1
        a = np.zeros(delay_samples + 1)
        a[0] = 1
        a[-1] = -decay
        effected = lfilter(b, a, buffer)
        effected = np.clip(effected, -32768, 32767)
        return effected.astype(np.int16)


class FDLConvolver:
    def __init__(self, ir: np.ndarray, block: int):
        """
        ir: (M, C_out), block: L
        Mono input. Output channels = ir.shape[1].
        """
        self.L = int(block)
        self.Nfft = 2 * self.L
        self.F = self.Nfft // 2 + 1
        self.C_out = ir.shape[1]

        # Partition IR and precompute spectra per output channel
        M = ir.shape[0]
        self.K = int(np.ceil(M / self.L))
        self.H = np.empty((self.C_out, self.K, self.F), dtype=np.complex64)
        for co in range(self.C_out):
            hpad = np.zeros(self.K * self.L, dtype=np.float32)
            hpad[:M] = ir[:, co]
            for k in range(self.K):
                seg = hpad[k * self.L:(k + 1) * self.L]
                self.H[co, k, :] = rfft(np.pad(seg, (0, self.L)))  # 2L FFT

        # State
        self.Xring = np.zeros((self.K, self.F), dtype=np.complex64)
        self.ridx = 0
        self.overlap = np.zeros((self.C_out, self.L), dtype=np.float32)

    def process_block(self, x_block_mono: np.ndarray, note_detector=None) -> np.ndarray:
        """
        x_block_mono: (L, 1) float32 in [-1,1]
        returns: (L, C_out) float32
        """
        x = x_block_mono[:, 0]
        X_i = rfft(np.pad(x, (0, self.L)))     # size 2L -> F bins

        if note_detector is not None:
            note_detector.update_spectrum(X_i)

        self.Xring[self.ridx, :] = X_i
        # Accumulate per output channel
        Yc = np.zeros((self.C_out, self.F), dtype=np.complex64)
        for co in range(self.C_out):
            Y = np.zeros(self.F, dtype=np.complex64)
            ridx = self.ridx
            for k in range(self.K):
                Y += self.H[co, k, :] * self.Xring[ridx, :]
                ridx = (ridx - 1) % self.K
            Yc[co, :] = Y

        # IFFT + OLA
        y = np.empty((self.C_out, self.L), dtype=np.float32)
        for co in range(self.C_out):
            y_time = irfft(Yc[co, :]).astype(np.float32)   # length 2L
            y_blk = y_time[:self.L] + self.overlap[co]
            self.overlap[co] = y_time[self.L:]
            y[co, :] = y_blk

        self.ridx = (self.ridx + 1) % self.K
        return y.T  # (L, C_out)


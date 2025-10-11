import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import lfilter

# === Audio effects ============================================================
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
    """
    Hybrid (head+tail) partitioned convolution:
      - Head: small time-domain FIR (low perceived latency)
      - Tail: FDL (uniform partitioned FFT, block size L)
    Mono input blocks (L,1) -> (L,C_out) float32.
    """

    # ---------- small helpers (added) ----------
    @staticmethod
    def _normalize_ir(ir: np.ndarray, peak=0.99) -> np.ndarray:
        ir = np.asarray(ir, dtype=np.float32)
        if ir.ndim == 1:
            ir = ir[:, None]
        m = np.max(np.abs(ir)) if ir.size else 1.0
        if m < 1e-12:
            return ir
        return (peak / m) * ir

    @staticmethod
    def _crossfade_split(ir: np.ndarray, head_len: int, xf_len: int = 32):
        """
        Split IR into head/tail with a small crossfade region to avoid a spectral kink.
        head taps = head_len; tail starts at head_len - xf_len, with complementary windows.
        """
        M, C = ir.shape
        head_len = int(max(0, min(head_len, M)))
        xf_len = int(max(0, min(xf_len, head_len)))

        if head_len == 0:
            head = np.zeros((0, C), dtype=np.float32)
            tail = ir.copy()
            return head, tail, 0  # tail starts at 0

        # windows on the overlap
        if xf_len > 0:
            n = np.arange(xf_len, dtype=np.float32)
            # raised-cosine from 1→0 for head, 0→1 for tail
            w_head = 0.5 * (1.0 + np.cos(np.pi * (n / (xf_len - 1))))
            w_tail = 1.0 - w_head
        else:
            w_head = None
            w_tail = None

        # head
        head = ir[:head_len, :].copy()
        if xf_len > 0:
            head[-xf_len:, :] *= w_head[:, None]

        # tail begins at (head_len - xf_len) so we keep continuity
        tail_start = max(0, head_len - xf_len)
        tail = ir[tail_start:, :].copy()
        if xf_len > 0 and tail.shape[0] >= xf_len:
            tail[:xf_len, :] *= w_tail[:, None]

        return head, tail, tail_start

    # ---------- constructor (unchanged signature) ----------
    def __init__(self, ir: np.ndarray, block: int):
        """
        ir: (M, C_out), block: L
        Mono input. Output channels = ir.shape[1].
        """
        # ---- Core sizes ----
        self.L = int(block)
        self.Nfft = 2 * self.L
        self.F = self.Nfft // 2 + 1

        # ---- Normalize IR to safe peak and set outputs ----
        ir = self._normalize_ir(ir, peak=0.99)
        M, self.C_out = ir.shape

        # ---- Choose head length (low-latency early reflections) ----
        # Default strategy: up to 256 taps or <= block size or <= M
        # You can change later via set_head_len()
        default_head = min(256, self.L, M)
        self.head_len = int(default_head)

        # Build head/tail (with crossfade)
        self._build_head_tail(ir)

    # ---------- public tuning hook (added) ----------
    def set_head_len(self, head_len: int, crossfade: int = 32):
        """
        Change low-latency head length at runtime.
        Will rebuild head/tail structures. Call between songs/IR loads.
        """
        self.head_len = int(max(0, head_len))
        # Rebuild from the original IR we cached:
        self._build_head_tail(self._ir_orig, crossfade=crossfade)

    # ---------- internal builders (added) ----------
    def _build_head_tail(self, ir: np.ndarray, crossfade: int = 32):
        """Create head FIR + tail FDL; preallocate work buffers/states."""
        # Cache original IR for potential retuning
        self._ir_orig = ir.copy()

        # Split IR
        head, tail, self.tail_start = self._crossfade_split(ir, self.head_len, xf_len=crossfade)
        self.h_head = head  # (H, C_out)
        self.tail_ir = tail  # (Mt, C_out)

        # --- Head state (time-domain FIR) ---
        H = self.h_head.shape[0]
        if H > 0:
            # per-channel streaming state for FIR: keep last H-1 input samples (mono input)
            self._head_xbuf = np.zeros(max(0, H - 1), dtype=np.float32)
            # preallocate head output buffer (C_out, L)
            self._y_head = np.zeros((self.C_out, self.L), dtype=np.float32)
        else:
            self._head_xbuf = None
            self._y_head = np.zeros((self.C_out, self.L), dtype=np.float32)

        # --- Tail FDL (uniform partitions with size L) ---
        Mt = self.tail_ir.shape[0]
        if Mt > 0:
            self.K = int(np.ceil(Mt / self.L))
            self.Hspec = np.empty((self.C_out, self.K, self.F), dtype=np.complex64)

            # Precompute tail partitions in freq domain (2L FFT each)
            self._part_buf = np.zeros(self.Nfft, dtype=np.float32)
            for co in range(self.C_out):
                # pad to K*L
                pad = np.zeros(self.K * self.L, dtype=np.float32)
                pad[:Mt] = self.tail_ir[:, co]
                # fill Hspec
                for k in range(self.K):
                    seg = pad[k * self.L:(k + 1) * self.L]
                    # place seg in first half; zero second half (already zero)
                    self._part_buf[:self.L] = seg
                    self._part_buf[self.L:] = 0.0
                    self.Hspec[co, k, :] = rfft(self._part_buf, n=self.Nfft).astype(np.complex64)

            # Tail state and work buffers
            self.Xring = np.zeros((self.K, self.F), dtype=np.complex64)  # newest at ridx
            self.ridx = 0
            self.overlap = np.zeros((self.C_out, self.L), dtype=np.float32)
            self._x2 = np.zeros(self.Nfft, dtype=np.float32)             # input 2L buffer
            self._X = np.zeros(self.F, dtype=np.complex64)
            self._Yc = np.zeros((self.C_out, self.F), dtype=np.complex64)
            self._y_time = np.zeros(self.Nfft, dtype=np.float32)
            self._y_tail = np.zeros((self.C_out, self.L), dtype=np.float32)
            self._order = np.arange(self.K, dtype=np.int32)              # for ring indexing
        else:
            # No tail → zero out FDL structures
            self.K = 0
            self.Hspec = None
            self.Xring = None
            self.ridx = 0
            self.overlap = np.zeros((self.C_out, self.L), dtype=np.float32)
            self._x2 = None
            self._X = None
            self._Yc = None
            self._y_time = None
            self._y_tail = np.zeros((self.C_out, self.L), dtype=np.float32)
            self._order = None

    # ---------- time-domain head (added) ----------
    def _process_head_time(self, x: np.ndarray):
        """
        Streaming FIR for the first head_len taps (per output channel).
        x: (L,) float32 mono block.
        writes into self._y_head (C_out, L).
        """
        H = self.h_head.shape[0]
        if H == 0:
            self._y_head.fill(0.0)
            return

        # Concatenate previous tail state and current block once
        if self._head_xbuf is not None and self._head_xbuf.size > 0:
            x_ext = np.concatenate((self._head_xbuf, x))  # length L + H - 1
        else:
            x_ext = x

        # Convolve per output channel (H is small → time-domain ok)
        # We only need the "valid" segment aligned with the current block:
        # y[n] = sum_{m=0..H-1} h[m]*x_ext[n+m], for n=0..L-1
        # Implement with np.convolve and slice:
        for co in range(self.C_out):
            y_full = np.convolve(x_ext, self.h_head[:, co], mode="full")  # len = len(x_ext)+H-1
            start = H - 1
            self._y_head[co, :] = y_full[start:start + self.L]

        # Update state: keep last H-1 of x_ext for next time
        if self._head_xbuf is not None and self._head_xbuf.size > 0:
            self._head_xbuf = x_ext[-(H - 1):].copy()

    # ---------- tail FDL (vectorized accumulation) ----------
    def _process_tail_fdl(self, x: np.ndarray):
        """
        x: (L,) float32 mono block.
        writes into self._y_tail (C_out, L).
        """
        if self.K == 0:
            self._y_tail.fill(0.0)
            return

        # Pack input into 2L buffer (no allocs)
        self._x2[:self.L] = x
        self._x2[self.L:] = 0.0

        # FFT
        self._X[:] = rfft(self._x2, n=self.Nfft).astype(np.complex64)

        # Ring update
        self.Xring[self.ridx, :] = self._X

        # Obtain newest..oldest view of Xring via np.take indices
        idx0 = self.ridx
        idxs = (idx0 - self._order) % self.K
        Xseq = np.take(self.Xring, idxs, axis=0)  # (K, F)

        # Vectorized accumulation over partitions:
        # Hspec: (C_out, K, F), Xseq: (K, F) -> Yc: (C_out, F)
        np.tensordot(self.Hspec, Xseq, axes=([1], [0]), out=self._Yc)

        # IFFT + OLA per channel
        for co in range(self.C_out):
            self._y_time[:] = irfft(self._Yc[co], n=self.Nfft).astype(np.float32)
            self._y_tail[co, :] = self._y_time[:self.L] + self.overlap[co]
            self.overlap[co, :] = self._y_time[self.L:]

        # advance ring
        self.ridx = (self.ridx + 1) % self.K

    # ---------- public API: unchanged ----------
    def process_block(self, x_block_mono: np.ndarray) -> np.ndarray:
        """
        x_block_mono: (L, 1) float32 in [-1,1]
        returns: (L, C_out) float32
        """
        # 1) head (low latency, time-domain)
        x = x_block_mono[:, 0].astype(np.float32, copy=False)
        self._process_head_time(x)

        # 2) tail (efficient FDL)
        self._process_tail_fdl(x)

        # 3) sum head + tail
        # buffers are (C_out, L) -> transpose to (L, C_out)
        y = (self._y_head + self._y_tail).T
        return y

class BypassFilter:
    """
    Simple pass-through filter for bypass mode.
    """
    def __init__(self):
        pass

    def process_block(self, x_block_mono: np.ndarray) -> np.ndarray:
        return x_block_mono
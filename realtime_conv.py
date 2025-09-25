import numpy as np
from scipy.fft import rfft, irfft
import scipy.io.wavfile as wav
import scipy.io as sio
import scipy.signal as signal
import numpy as np
import time

class PartitionedConvolver:
    def __init__(self, impulse_response: np.ndarray, sector_size: int):
        """
        Initialize partitioned overlap-add convolver.

        Parameters:
        - impulse_response: array-like, the impulse response h[n]
        - sector_size: int, number of samples per sector (block)
        """
        self.sector_size = sector_size
        self.fft_size = 2 * sector_size  # for circular convolution safety

        # Partition the impulse response
        h = np.array(impulse_response, dtype=float)
        num_parts = int(np.ceil(len(h) / sector_size))
        self.H_parts = []
        for i in range(num_parts):
            part = h[i*sector_size:(i+1)*sector_size]
            part_padded = np.zeros(self.fft_size)
            part_padded[:len(part)] = part
            self.H_parts.append(rfft(part_padded))

        # Buffers
        self.input_fft_buffer = []
        self.overlap = np.zeros(sector_size)

    def process(self, x_block: np.ndarray) -> np.ndarray:
        """
        Process one sector of input and return the corresponding sector of output.

        Parameters:
        - x_block: array-like, length == sector_size

        Returns:
        - y_block: array of length == sector_size
        """
        if len(x_block) != self.sector_size:
            raise ValueError("Input block must have length equal to sector_size")

        # FFT of the input block (zero-padded)
        x_padded = np.zeros(self.fft_size)
        x_padded[:self.sector_size] = x_block
        X = rfft(x_padded)

        # Store input spectrum at the front of the buffer
        self.input_fft_buffer.insert(0, X)
        if len(self.input_fft_buffer) > len(self.H_parts):
            self.input_fft_buffer.pop()

        # Convolution in frequency domain with all partitions
        Y_accum = np.zeros(self.fft_size)
        for k, Hk in enumerate(self.H_parts):
            if k < len(self.input_fft_buffer):
                Y_accum += irfft(self.input_fft_buffer[k] * Hk, self.fft_size)

        # Add overlap and extract current block
        y_block = Y_accum[:self.sector_size] + self.overlap

        # Save new overlap for next call
        self.overlap = Y_accum[self.sector_size:]

        return y_block

    def process_stream(self, signal: np.ndarray) -> np.ndarray:
        """
        Process an entire signal stream by splitting it into sectors automatically.

        Parameters:
        - signal: array-like, input signal of arbitrary length

        Returns:
        - output: array-like, convolved signal (same length as input)
        """
        signal = np.array(signal, dtype=float)
        n = len(signal)
        output = []

        for i in range(0, n, self.sector_size):
            block = signal[i:i+self.sector_size]
            if len(block) < self.sector_size:
                # Zero-pad last block
                block = np.pad(block, (0, self.sector_size - len(block)))
            y_block = self.process(block)
            # Trim padding if on last block
            if i + self.sector_size > n:
                y_block = y_block[:n - i]
            output.append(y_block)

        return np.concatenate(output)


# Example usage
if __name__ == "__main__":
    # h = np.random.randn(8096)  # long impulse response
    ir_mat_file = "air_binaural_aula_carolina_0_1_1_90_3.mat"
    ir_key = "h_air"
    mat_data = sio.loadmat(ir_mat_file)
    if ir_key not in mat_data:
        raise ValueError(f"Impulse response key '{ir_key}' not found in {ir_mat_file}")
    h = mat_data[ir_key].squeeze().astype(np.float32)

    sector_size = 1024
    convolver = PartitionedConvolver(h, sector_size)
    
    # signal = np.random.randn(1024)
    wav_file = "TestAudio.wav"
    fs_wav, signal = wav.read(wav_file)
    signal = signal.astype(np.float32)
    

    # Real-time style block processing
    print("--- Real-time blocks ---")
    for i in range(0, sector_size*100, sector_size): #len(signal)
        
        start = time.perf_counter()

        block = signal[i:i+sector_size]
        if len(block) < sector_size:
            block = np.pad(block, (0, sector_size - len(block)))
        y = convolver.process(block)

        end = time.perf_counter()
        print(f"Execution time: {end - start:.6f} seconds")

        print(f"Block {i//sector_size}: {y}")

    # Full stream processing
    # print("\n--- Full stream ---")
    # convolver2 = PartitionedConvolver(h, sector_size)
    # y_stream = convolver2.process_stream(signal)
    # print(y_stream)

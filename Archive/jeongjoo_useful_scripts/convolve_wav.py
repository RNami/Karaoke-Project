import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

def convolve_two_wavs(input_wav, ir_wav, output_file):
    """
    Convolves two wav files (input signal * impulse response).

    Args:
        input_wav (str): Path to the input wav file.
        ir_wav (str): Path to the impulse response wav file.
        output_file (str): Path to save the output wav file.
    """
    # Load input wav
    fs_in, audio = wav.read(input_wav)
    audio = audio.astype(np.float32)

    # Load IR wav
    fs_ir, ir = wav.read(ir_wav)
    ir = ir.astype(np.float32)

    # Ensure sample rates match
    if fs_in != fs_ir:
        raise ValueError(f"Sample rate mismatch: input={fs_in} Hz, IR={fs_ir} Hz")

    # Handle stereo by taking just the first channel (or adapt to stereo convolution)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if ir.ndim > 1:
        ir = ir[:, 0]

    # Convolve using FFT convolution
    convolved = signal.fftconvolve(audio, ir, mode="full")

    # Normalize to avoid clipping
    convolved = convolved / np.max(np.abs(convolved)) * 0.9

    # Convert back to int16
    convolved = (convolved * 32767).astype(np.int16)

    # Save result
    wav.write(output_file, fs_in, convolved)

# Example usage:
convolve_two_wavs("spongebob.wav", "ia_stnikolaus.wav", "output.wav")

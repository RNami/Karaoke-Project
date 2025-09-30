import scipy.io.wavfile as wav
import scipy.io as sio
import scipy.signal as signal
import numpy as np

def convolve_wav_with_ir(wav_file, ir_mat_file, ir_key, output_file):
    """
    Convolves a .wav file with an impulse response stored in a .mat file.

    Args:
        wav_file (str): Path to input wav file.
        ir_mat_file (str): Path to .mat file containing the impulse response.
        ir_key (str): Key of the impulse response variable in the .mat file.
        output_file (str): Path to save the output wav file.
    """
    # Load input wav
    fs_wav, audio = wav.read(wav_file)
    audio = audio.astype(np.float32)

    # Load impulse response from .mat file
    mat_data = sio.loadmat(ir_mat_file)
    if ir_key not in mat_data:
        raise ValueError(f"Impulse response key '{ir_key}' not found in {ir_mat_file}")
    ir = mat_data[ir_key].squeeze().astype(np.float32)

    # Make sure sample rates match
    fs_ir = 48000  # given IR is at 48 kHz
    if fs_wav != fs_ir:
        raise ValueError(f"Sample rate mismatch: WAV file is {fs_wav} Hz but IR is {fs_ir} Hz")

    # Convolve (FFT-based for efficiency)
    convolved = signal.fftconvolve(audio, ir, mode='full')

    # Normalize to prevent clipping
    convolved = convolved / np.max(np.abs(convolved)) * 0.9

    # Convert back to int16 for WAV
    convolved = (convolved * 32767).astype(np.int16)

    # Save output
    wav.write(output_file, fs_wav, convolved)

# Example usage
convolve_wav_with_ir("TestAudio.wav", "air_binaural_aula_carolina_0_1_1_90_3.mat", "h_air", "output.wav")

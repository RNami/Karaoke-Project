import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

def plot_impulse_response(ir_wav_file):
    """
    Loads an impulse response from a .wav file and plots it.

    Args:
        ir_wav_file (str): Path to .wav file containing the impulse response.
    """
    # Load impulse response
    fs, ir = wav.read(ir_wav_file)

    # If stereo, take one channel
    if ir.ndim > 1:
        ir = ir[:, 0]

    # Convert to float (normalize if integer type)
    ir = ir.astype(np.float32)
    if np.max(np.abs(ir)) > 0:
        ir = ir / np.max(np.abs(ir))

    # Create time axis
    t = np.arange(len(ir)) / fs

    # Debug info
    print("Sampling Rate:", fs)
    print("Max amplitude:", np.max(np.abs(ir)))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, ir, color="blue")
    plt.title("Impulse Response from WAV")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (normalized)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
plot_impulse_response("ia_stnikolaus.wav")

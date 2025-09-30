import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def plot_impulse_response(ir_mat_file, ir_key, fs=48000):
    """
    Loads an impulse response from a .mat file and plots it.

    Args:
        ir_mat_file (str): Path to .mat file containing the impulse response.
        ir_key (str): Key of the impulse response variable in the .mat file.
        fs (int): Sampling frequency (default 48 kHz).
    """
    # Load impulse response
    mat_data = sio.loadmat(ir_mat_file)
    if ir_key not in mat_data:
        raise ValueError(f"Impulse response key '{ir_key}' not found in {ir_mat_file}")
    ir = mat_data[ir_key].squeeze()

    # Create time axis
    t = np.arange(len(ir)) / fs
    print(max(abs(ir)))
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, ir, color="blue")
    plt.title("Impulse Response")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
plot_impulse_response("air_binaural_aula_carolina_0_1_1_90_3.mat", "h_air")

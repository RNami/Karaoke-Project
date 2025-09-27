# io.py
import sys
import pyaudio

BLOCK = 256  # window size in samples

def list_devices(p: pyaudio.PyAudio):
    """
    Returns (inputs, outputs) filtered so that:
      • On Windows  -> only WASAPI devices.
      • Inputs      -> microphones only.
      • Outputs     -> speakers only.
    """
    inputs, outputs = [], []

    # Detect Windows
    is_windows = sys.platform.startswith("win")

    # Get the index of the WASAPI host API if present
    wasapi_index = None
    if is_windows:
        for i in range(p.get_host_api_count()):
            api_info = p.get_host_api_info_by_index(i)
            if api_info.get("name", "").lower().startswith("windows wasapi"):
                wasapi_index = i
                break

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # --- Host API filter (Windows: WASAPI only) ---
        if is_windows and wasapi_index is not None:
            if info.get("hostApi") != wasapi_index:
                continue  # skip non-WASAPI devices

        name     = info.get("name", f"Device {i}")
        max_in   = info.get("maxInputChannels", 0)
        max_out  = info.get("maxOutputChannels", 0)
        rate     = int(info.get("defaultSampleRate", 0))

        # --- Microphone filter for inputs ---
        if max_in > 0:
            # heuristic: name contains typical mic keywords
            if any(k in name.lower() for k in ["mic", "microphone", "input", "line in"]):
                inputs.append((i, name, max_in, rate))

        # --- Speaker filter for outputs ---
        if max_out > 0:
            # heuristic: name contains typical speaker keywords
            if any(k in name.lower() for k in ["speaker", "headphone", "out", "output"]):
                outputs.append((i, name, max_out, rate))

    return inputs, outputs

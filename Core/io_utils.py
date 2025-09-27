import os

def get_wasapi_devices(pa):
    """Return only WASAPI devices (mic / speaker / loopback)."""
    devices = []
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        hostapi_name = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        if "WASAPI" not in hostapi_name and os.name == "nt":
            continue
        devices.append({
            "index": idx,
            "name": info["name"],
            "max_in": int(info["maxInputChannels"]),
            "max_out": int(info["maxOutputChannels"]),
            "default_sr": int(info["defaultSampleRate"]),
        })
    return devices
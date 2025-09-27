# io.py
import pyaudio

BLOCK = 256  # window size in samples

def list_devices(p: pyaudio.PyAudio):
    inputs, outputs = [], []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", f"Device {i}")
        max_in  = info.get("maxInputChannels", 0)
        max_out = info.get("maxOutputChannels", 0)
        rate    = int(info.get("defaultSampleRate", 0))
        if max_in  > 0:  inputs.append((i, name, max_in, rate))
        if max_out > 0: outputs.append((i, name, max_out, rate))
    return inputs, outputs

def choose_device_index(options, prompt):
    print(prompt)
    for idx, name, ch, fs in options:
        print(f"  [{idx}] {name}  ch:{ch}  fs:{fs}")
    raw = input("Enter index (blank for default): ").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None

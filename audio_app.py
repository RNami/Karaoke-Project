import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from audio_engine import AudioEngine, get_wasapi_devices, COMMON_SAMPLE_RATES, BUFFER_SIZES, PREFERRED_CHANNELS

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Karaoke Reverb GUI")
        self.engine = AudioEngine()
        self.stream_thread = None

        self.devices = get_wasapi_devices(self.engine.pa)
        self.input_devices = [d for d in self.devices if d["max_in"] > 0]
        self.output_devices = [d for d in self.devices if d["max_out"] > 0]

        ttk.Label(root, text="Input Device:").grid(row=0, column=0, sticky="e")
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(root, textvariable=self.input_var, width=50, state="readonly")
        self.input_combo['values'] = [f"[{d['index']}] {d['name']}" for d in self.input_devices]
        self.input_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(root, text="Output Device:").grid(row=1, column=0, sticky="e")
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(root, textvariable=self.output_var, width=50, state="readonly")
        self.output_combo['values'] = [f"[{d['index']}] {d['name']}" for d in self.output_devices]
        self.output_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(root, text="Sample Rate:").grid(row=2, column=0, sticky="e")
        self.sr_var = tk.StringVar(value=str(COMMON_SAMPLE_RATES[0]))
        self.sr_combo = ttk.Combobox(root, textvariable=self.sr_var, values=[str(sr) for sr in COMMON_SAMPLE_RATES], state="readonly")
        self.sr_combo.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(root, text="Channels:").grid(row=3, column=0, sticky="e")
        self.ch_var = tk.StringVar(value=str(PREFERRED_CHANNELS[0]))
        self.ch_combo = ttk.Combobox(root, textvariable=self.ch_var, values=[str(ch) for ch in PREFERRED_CHANNELS], state="readonly")
        self.ch_combo.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(root, text="Buffer Size:").grid(row=4, column=0, sticky="e")
        self.buf_var = tk.StringVar(value=str(BUFFER_SIZES[0]))
        self.buf_combo = ttk.Combobox(root, textvariable=self.buf_var, values=[str(b) for b in BUFFER_SIZES], state="readonly")
        self.buf_combo.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(root, text="Effect:").grid(row=5, column=0, sticky="e")
        self.effect_var = tk.StringVar(value="None")
        self.effect_combo = ttk.Combobox(root, textvariable=self.effect_var, values=["None", "Robot Voice", "Concert Hall"], state="readonly")
        self.effect_combo.grid(row=5, column=1, padx=5, pady=5)

        self.start_btn = ttk.Button(root, text="Start", command=self.start_audio)
        self.start_btn.grid(row=6, column=0, padx=5, pady=10)
        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_audio, state="disabled")
        self.stop_btn.grid(row=6, column=1, padx=5, pady=10)

    def start_audio(self):
        try:
            inp_idx = self.input_combo.current()
            out_idx = self.output_combo.current()
            if inp_idx == -1 or out_idx == -1:
                messagebox.showerror("Error", "Please select both input and output devices.")
                return
            inp = self.input_devices[inp_idx]
            out = self.output_devices[out_idx]
            rate = int(self.sr_var.get())
            ch = int(self.ch_var.get())
            buf = int(self.buf_var.get())
            effect = self.effect_var.get()

            self.engine.start_stream(inp["index"], out["index"], rate, ch, buf, effect)
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.stream_thread = threading.Thread(target=self.monitor_stream, daemon=True)
            self.stream_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start audio stream:\n{e}")

    def stop_audio(self):
        self.engine.stop_stream()
        # Only update GUI if called from main thread
        if threading.current_thread() == threading.main_thread():
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def monitor_stream(self):
        while self.engine.running and self.engine.stream is not None and self.engine.stream.is_active():
            time.sleep(0.1)
        # Schedule GUI update on main thread``````````````
        self.root.after(0, self.stop_audio)

    def on_close(self):
        self.engine.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
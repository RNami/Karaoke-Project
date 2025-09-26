#audio_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time

from audio_engine import AudioEngine, get_wasapi_devices


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Karaoke Reverb GUI (Two-Stream)")
        self.engine = AudioEngine()

        # ---- Device selection -------------------------------------------------
        self.devices = get_wasapi_devices(self.engine.pa)
        self.input_devices = [d for d in self.devices if d["max_in"] > 0]
        self.output_devices = [d for d in self.devices if d["max_out"] > 0]

        ttk.Label(root, text="Input Device:").grid(row=0, column=0, sticky="e")
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(root, textvariable=self.in_var,
                                     values=[f"[{d['index']}] {d['name']}" for d in self.input_devices],
                                     state="readonly", width=50)
        self.in_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(root, text="Output Device:").grid(row=1, column=0, sticky="e")
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(root, textvariable=self.out_var,
                                      values=[f"[{d['index']}] {d['name']}" for d in self.output_devices],
                                      state="readonly", width=50)
        self.out_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(root, text="Effect:").grid(row=2, column=0, sticky="e")
        self.effect_var = tk.StringVar(value="None")
        self.effect_combo = ttk.Combobox(root, textvariable=self.effect_var,
                                         values=["None", "Robot Voice", "Concert Hall"],
                                         state="readonly")
        self.effect_combo.grid(row=2, column=1, padx=5, pady=5)

        # Level meter
        ttk.Label(root, text="Mic Level:").grid(row=3, column=0, sticky="e")
        self.level = tk.DoubleVar()
        self.level_bar = ttk.Progressbar(root, variable=self.level,
                                         maximum=100, length=300)
        self.level_bar.grid(row=3, column=1, padx=5, pady=5)

        # Start/Stop
        self.start_btn = ttk.Button(root, text="Start", command=self.start_audio)
        self.start_btn.grid(row=4, column=0, padx=5, pady=10)
        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_audio, state="disabled")
        self.stop_btn.grid(row=4, column=1, padx=5, pady=10)

    # -------------------------------------------------------------------------
    def start_audio(self):
        try:
            i_idx = self.in_combo.current()
            o_idx = self.out_combo.current()
            if i_idx == -1 or o_idx == -1:
                messagebox.showerror("Error", "Select both input and output devices.")
                return
            inp = self.input_devices[i_idx]
            out = self.output_devices[o_idx]
            effect = self.effect_var.get()

            self.engine.start_stream(inp["index"], out["index"], effect)
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # GUI updates
            self.update_level_bar()
            threading.Thread(target=self.monitor_stream, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start audio:\n{e}")

    def stop_audio(self):
        try:
            self.engine.stop_stream()  # idempotent
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Could not stop audio:\n{e}")


    def monitor_stream(self):
        while self.engine.running:
            in_stream = self.engine.in_stream  # local copy
            if in_stream is None:
                break
            if not in_stream.is_active():
                break
            time.sleep(0.05)
        self.root.after(0, lambda: (
            self.start_btn.config(state="normal"),
            self.stop_btn.config(state="disabled")
        ))


    def update_level_bar(self):
        self.level.set(self.engine.current_level)
        if self.engine.running:
            self.root.after(50, self.update_level_bar)

    def on_close(self):
        self.engine.running = False
        self.engine.stop_stream()
        self.engine.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

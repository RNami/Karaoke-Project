# audio_app.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os

from note_detection.audio_engine import AudioEngine, get_wasapi_devices


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
                                         values=["None", "Robot Voice", "Concert Hall", "Convolver"],
                                         state="readonly")
        self.effect_combo.grid(row=2, column=1, padx=5, pady=5)
        self.effect_combo.bind("<<ComboboxSelected>>", self.on_effect_changed)

        # IR selection row
        ttk.Label(root, text="IR File:").grid(row=3, column=0, sticky="e")
        self.ir_path_var = tk.StringVar(value="(none)")
        self.ir_label = ttk.Label(root, textvariable=self.ir_path_var, width=50, anchor="w")
        self.ir_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.ir_btn = ttk.Button(root, text="Choose IR...", command=self.choose_ir)
        self.ir_btn.grid(row=3, column=2, padx=5, pady=5)

        # Level meter
        ttk.Label(root, text="Mic Level:").grid(row=4, column=0, sticky="e")
        self.level = tk.DoubleVar()
        self.level_bar = ttk.Progressbar(root, variable=self.level,
                                         maximum=100, length=300)
        self.level_bar.grid(row=4, column=1, padx=5, pady=5)

        #current pitch# inside your GUI setup
        self.note = tk.StringVar()

        self.note_label = ttk.Label(root, text="Current Pitch:")
        self.note_label.grid(row=5, column=0, sticky="e")

        self.note_val = ttk.Label(root, textvariable=self.note)
        self.note_val.grid(row=5, column=1, sticky="w")


        # Start/Stop
        self.start_btn = ttk.Button(root, text="Start", command=self.start_audio)
        self.start_btn.grid(row=6, column=0, padx=5, pady=10)
        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_audio, state="disabled")
        self.stop_btn.grid(row=6, column=1, padx=5, pady=10)

    # -------------------------------------------------------------------------
    def choose_ir(self):
        path = filedialog.askopenfilename(
            title="Select IR file",
            filetypes=[
                ("Impulse Responses", ("*.mat", "*.wav", "*.flac", "*.aiff", "*.aif")),
                ("All files", "*")
            ]
        )
        if not path:
            return
        # store full path for engine, show only filename in label
        self.engine.ir_path = path
        self.ir_path_var.set(os.path.basename(path))

        # if sample rate is already known, load immediately
        if getattr(self.engine, "in_rate", None):
            try:
                self.engine.load_ir(path, target_fs=self.engine.in_rate)
                messagebox.showinfo(
                    "IR loaded", f"Loaded IR and resampled to {self.engine.in_rate} Hz."
                )
            except Exception as e:
                messagebox.showerror("IR load error", f"Could not load IR:\n{e}")
                self.engine.convolver = None


    def on_effect_changed(self, event=None):
        eff = self.effect_var.get()
        if eff == "Convolver" and (self.engine.convolver is None and not self.engine.ir_path):
            # no IR chosen yet -> prompt
            res = messagebox.askyesno("Convolver selected", "No IR loaded. Do you want to choose an IR file now?")
            if res:
                self.choose_ir()

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

            # if convolver selected and IR path is set but not loaded, load it at input fs
            if effect == "Convolver" and self.engine.convolver is None and self.engine.ir_path is not None:
                try:
                    # we don't yet know in_rate until start_stream inspects the device - so load after opening stream
                    # the engine.start_stream will try to load if engine.ir_path is set (it tries to load at start)
                    pass
                except Exception as e:
                    messagebox.showerror("IR load error", f"Could not load IR:\n{e}")
                    return

            self.engine.start_stream(inp["index"], out["index"], effect)
            # If effect is Convolver and convolver wasn't initialized, try loading now using discovered in_rate
            if effect == "Convolver" and self.engine.convolver is None and self.engine.ir_path:
                try:
                    self.engine.load_ir(self.engine.ir_path, target_fs=self.engine.in_rate)
                except Exception as e:
                    messagebox.showwarning("IR warning", f"Failed to initialize convolver IR:\n{e}")

            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            # GUI updates
            self.update_level_bar()
            self.update_pitch_label()
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

        
    def update_pitch_label(self):
        self.note.set(self.engine.current_note)
        if self.engine.running:
            self.root.after(50, self.update_pitch_label)



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

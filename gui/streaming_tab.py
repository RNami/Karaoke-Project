import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

class StreamingTab:
    def __init__(self, notebook, engine, devices):
        self.engine = engine
        self.frame = ttk.Frame(notebook)

        # ----------------------------
        # Device selection
        # ----------------------------
        self.devices = devices

        self.input_devices = [d for d in devices if d["max_in"] > 0]
        self.output_devices = [d for d in devices if d["max_out"] > 0]

        ttk.Label(self.frame, text="Input Device:").grid(row=0, column=0)
        self.in_var = tk.StringVar()
        
        self.in_combo = ttk.Combobox(
            self.frame, textvariable=self.in_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.input_devices],
            state="readonly", width=50
        )
        self.in_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.frame, text="Output Device:").grid(row=1, column=0)
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(
            self.frame, textvariable=self.out_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.output_devices],
            state="readonly", width=50
        )
        self.out_combo.grid(row=1, column=1)

        # ----------------------------
        # Effect selection
        # ----------------------------
        
        ttk.Label(self.frame, text="Effect:").grid(row=2, column=0)
        self.effect_var = tk.StringVar(value="None")
        self.effect_combo = ttk.Combobox(
            self.frame, textvariable=self.effect_var,
            values=["None", "Robot Voice", "Concert Hall", "Convolver"],
            state="readonly"
        )
        self.effect_combo.grid(row=2, column=1)
        self.effect_combo.bind("<<ComboboxSelected>>", self.on_effect_changed)

        # ----------------------------
        # IR file chooser
        # ----------------------------
        ttk.Label(self.frame, text="IR File:").grid(row=3, column=0)
        self.ir_path_var = tk.StringVar(value="(none)")
        self.ir_label = ttk.Label(self.frame, textvariable=self.ir_path_var, width=50, anchor="w")
        self.ir_label.grid(row=3, column=1)
        ttk.Button(self.frame, text="Choose IR...", command=self.choose_ir).grid(row=3, column=2)

        # ----------------------------
        # Mic level
        # ----------------------------
        ttk.Label(self.frame, text="Mic Level:").grid(row=4, column=0)
        self.level = tk.DoubleVar()
        self.level_bar = ttk.Progressbar(self.frame, variable=self.level, maximum=100, length=300)
        self.level_bar.grid(row=4, column=1)

        # ----------------------------
        # Pitch
        # ----------------------------
        self.note = tk.StringVar()
        ttk.Label(self.frame, text="Current Pitch:").grid(row=5, column=0)
        ttk.Label(self.frame, textvariable=self.note).grid(row=5, column=1)

        # ----------------------------
        # Start/Stop
        # ----------------------------
        self.start_btn = ttk.Button(self.frame, text="Start", command=self.start_audio)
        self.start_btn.grid(row=6, column=0)
        self.stop_btn = ttk.Button(self.frame, text="Stop", command=self.stop_audio, state="disabled")
        self.stop_btn.grid(row=6, column=1)



    def choose_ir(self):
        path = filedialog.askopenfilename(
            title="Select IR file",
            filetypes=[("Impulse Responses", ("*.mat", "*.wav", "*.flac", "*.aiff", "*.aif")), ("All files", "*")]
        )
        if not path:
            return
        self.engine.ir_path = path
        self.ir_path_var.set(os.path.basename(path))

    def on_effect_changed(self, event=None):
        if self.effect_var.get() == "Convolver" and not self.engine.ir_path:
            if messagebox.askyesno("Convolver selected", "No IR loaded. Do you want to choose one?"):
                self.choose_ir()

    def start_audio(self):
        try:
            i_idx, o_idx = self.in_combo.current(), self.out_combo.current()
            if i_idx == -1 or o_idx == -1:
                messagebox.showerror("Error", "Select input and output devices.")
                return
            inp, out = self.input_devices[i_idx], self.output_devices[o_idx]
            self.engine.start_stream(inp["index"], out["index"], self.effect_var.get())
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.update_level_bar()
            self.update_pitch_label()
            threading.Thread(target=self.monitor_stream, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not start audio:\n{e}")

    def stop_audio(self):
        self.engine.stop_stream()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def monitor_stream(self):
        while self.engine.running:
            if not self.engine.in_stream or not self.engine.in_stream.is_active():
                break
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def update_level_bar(self):
        self.level.set(self.engine.current_level)
        if self.engine.running:
            self.frame.after(50, self.update_level_bar)

    def update_pitch_label(self):
        self.note.set(self.engine.current_note)
        if self.engine.running:
            self.frame.after(50, self.update_pitch_label)

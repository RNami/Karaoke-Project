import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

from gui.widgets import (
    make_label,
    make_button,
    make_combobox,
    make_progressbar,
    make_frame,
    configure_grid
)

class StreamingTab:
    def __init__(self, notebook, engine, devices):
        self.engine = engine
        self.frame = make_frame(notebook)

        # Configure grid for resizing
        # 3 columns (labels, controls, optional buttons)
        configure_grid(self.frame, cols=3, rows=8, weight=1)

        # ----------------------------
        # Device selection
        # ----------------------------
        self.devices = devices
        self.input_devices = [d for d in devices if d["max_in"] > 0]
        self.output_devices = [d for d in devices if d["max_out"] > 0]

        make_label(
            self.frame, text="Input Device:",
            row=0, column=0, sticky="e"
        )
        self.in_var = tk.StringVar()
        self.in_combo = make_combobox(
            self.frame, textvariable=self.in_var,
            values=[f"[{d['index']} {d['name']}" for d in self.input_devices],
            state="readonly", width=50, row=0, column=1, sticky="we"
        )

        make_label(
            self.frame, text="Output Device:",
            row=1, column=0, sticky="e"
        )
        self.out_var = tk.StringVar()
        self.out_combo = make_combobox(
            self.frame, textvariable=self.out_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.output_devices],
            state="readonly", width=50, row=1, column=1, sticky="we"
        )

        # ----------------------------
        # Effect selection
        # ----------------------------
        make_label(
            self.frame, text="Effect:",
            row=2, column=0, sticky="e"
        )
        self.effect_var = tk.StringVar(value="None")
        self.effect_combo = make_combobox(
            self.frame, textvariable=self.effect_var,
            values = ["None", "Robot Voice", "Concert Hall", "Convolver"],
            state="readonly", row=2, column=1, sticky="we"
        )
        self.effect_combo.bind("<<ComboboxSelected>>", self.on_effect_changed)

        # ----------------------------
        # IR file chooser
        # ----------------------------
        make_label(
            self.frame, text="IR File:",
            row=3, column=0, sticky="e"
        )
        self.ir_path_var = tk.StringVar(value="(none)")
        make_label(
            self.frame, textvariable=self.ir_path_var,
            row=3, column=1, sticky="we"
        )
        make_button(
            self.frame, "Choose IR...",
            command=self.choose_ir,
            row=3, column=2,
        )

        # ----------------------------
        # Mic level
        # ----------------------------
        make_label(
            self.frame, text="Mic Level:",
            row=4, column=0, sticky="e"
        )
        self.level = tk.DoubleVar()
        make_progressbar(
            self.frame, variable=self.level,
            maximum=100, length=300,
            row=4, column=1, sticky="we"
        )

        # ----------------------------
        # Pitch display
        # ----------------------------
        self.note = tk.StringVar()
        make_label(
            self.frame, text="Current Pitch:",
            row=5, column=0, sticky="e"
        )
        make_label(
            self.frame, textvariable=self.note,
            row=5, column=1, sticky="w"
        )

        # ----------------------------
        # Start/Stop buttons
        # ----------------------------
        self.start_btn = make_button(
            self.frame, text="Start",
            command=self.start_audio,
            row=6, column=0, sticky="we",
        )
        self.stop_btn = make_button(
            self.frame, text="Stop",
            command=self.stop_audio,
            row=6, column=1, sticky="we",
            state="disabled"
        )


    # ----------------------------
    # IR file chooser
    # ----------------------------
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
        self.engine.ir_path = path
        self.ir_path_var.set(os.path.basename(path))

    # ----------------------------
    # Effect selection handler
    # ----------------------------
    def on_effect_changed(self, event=None):
        if self.effect_var.get() == "Convolver" and not self.engine.ir_path:
            if messagebox.askyesno("Convolver selected", "No IR loaded. Do you want to choose one?"):
                self.choose_ir()

    # ----------------------------
    # Stream controls
    # ----------------------------
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

    # ----------------------------
    # UI updates
    # ----------------------------
    def update_level_bar(self):
        self.level.set(self.engine.current_level)
        if self.engine.running:
            self.frame.after(50, self.update_level_bar)

    def update_pitch_label(self):
        self.note.set(self.engine.current_note)
        if self.engine.running:
            self.frame.after(50, self.update_pitch_label)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

from utils.rir.rir_record import RIRRecorder
from gui.base_widgets import (
    make_label,
    make_entry,
    make_button,
    make_textbox,
    make_frame,
    configure_grid,
    make_combobox,
)


class RIRTab:
    def __init__(self, notebook, engine, devices):
        self.engine = engine
        self.frame = make_frame(notebook)

        # ----------------------------
        # Responsive grid layout
        # ----------------------------
        configure_grid(self.frame, rows=[0, 1, 2], cols=[0, 1, 2], weight=1)


        # ----------------------------
        # Device selection (input/output)
        # ----------------------------
        self.input_devices = [d for d in devices if d["max_in"] > 0]
        self.output_devices = [d for d in devices if d["max_out"] > 0]

        make_label(self.frame, "Input Device:", row=0, column=0, sticky="e")
        self.input_var = tk.StringVar()
        self.input_combo = make_combobox(
            self.frame,
            textvariable=self.input_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.input_devices],
            state="readonly",
            width=50,
            row=0,
            column=1,
            sticky="we"
        )
        if self.input_devices:
            self.input_combo.current(0)

        make_label(self.frame, "Output Device:", row=1, column=0, sticky="e")
        self.output_var = tk.StringVar()
        self.output_combo = make_combobox(
            self.frame,
            textvariable=self.output_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.output_devices],
            state="readonly",
            width=50,
            row=1,
            column=1,
            sticky="we"
        )
        if self.output_devices:
            self.output_combo.current(0)


        # ----------------------------
        # Save path selection
        # ----------------------------
        make_label(
            self.frame,
            "Save RIR As:",
            row=2, column=0, sticky="e",
        )
        self.rir_save_label_var = tk.StringVar(value="(none)")
        make_label(
            self.frame,
            textvariable=self.rir_save_label_var,
            row=2, column=1, sticky="w",
        )
        make_button(
            self.frame,
            "Browse RIR File...",
            command=self.choose_save_file,
            row=2, column=2, sticky="we",
        )

        # ----------------------------
        # Sweep file selection
        # ----------------------------
        make_label(
            self.frame,
            "Sweep File:",
            row=3, column=0, sticky="e",
        )
        self.sweep_file_path = 'Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav'
        self.sweep_file_label_var = tk.StringVar(
            value=os.path.basename(self.sweep_file_path)
        )
        make_label(
            self.frame,
            textvariable=self.sweep_file_label_var,
            row=3, column=1, sticky="w",
        )
        make_button(
            self.frame,
            "Browse Sweep File...",
            command=self.choose_sweep_file,
            row=3, column=2, sticky="we",
        )

        # ----------------------------
        # Record length input
        # ----------------------------
        make_label(
            self.frame,
            text="Record Length (sec):",
            row=4, column=0, sticky="e"
        )
        self.record_length_var = tk.StringVar(value="20")
        make_entry(
            self.frame,
            textvariable=self.record_length_var,
            width=5,
            row=4, column=1, sticky="we"
        )
        
        # ------------------------------------------------------------------
        # Measure button
        # ------------------------------------------------------------------
        make_button(self.frame,
                    "Measure RIR",
                    command=self.start_rir_measurement,
                    row=4, column=2, sticky="we")
        
        # ------------------------------------------------------------------
        # Log box (read-only)
        # ------------------------------------------------------------------
        make_label(self.frame,
                   "Log:",
                   row=5, column=0,
                   columnspan=2,
                   sticky="ne")
        self.log_box = make_textbox(self.frame,
                                    height=12, width=70,
                                    row=5, column=1, columnspan=2, sticky="nsew")
        self.log_box.config(state="disabled")  # make read-only
    # ----------------------------------------------------------------------
    # Logging helper
    # ----------------------------------------------------------------------
    def _log(self, msg: str):
        self.log_box.config(state="normal")    # temporarily allow edits
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")  # set back to read-only

    # ----------------------------------------------------------------------
    # File chooser
    # ----------------------------------------------------------------------
    def choose_save_file(self):
        path = filedialog.asksaveasfilename(
            title="Save measured RIR as...",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if path:
            self.rir_save_path = path
            self.rir_save_label_var.set(os.path.basename(path))

    # ----------------------------------------------------------------------
    # Measurement logic
    # ----------------------------------------------------------------------
    def start_rir_measurement(self):
        # --------------------------
        # Parse record length
        # --------------------------
        try:
            record_length = int(self.record_length_var.get())
        except ValueError:
            messagebox.showwarning("Invalid record length", "Using default of 20 sec.")
            record_length = 20

        # --------------------------
        # Preconditions
        # --------------------------
        if not hasattr(self, "rir_save_path") or not self.rir_save_path:
            messagebox.showwarning("Save path", "Select a file path to save the measured RIR first.")
            return
        
        if not hasattr(self, "sweep_file_path") or not self.sweep_file_path:
            messagebox.showwarning("Sweep file", "Select a sweep file first.")
            return

        # --------------------------
        # Map selected device names to indices
        # --------------------------
        input_index = self.input_devices[self.input_combo.current()]["index"]
        output_index = self.output_devices[self.output_combo.current()]["index"]

        # --------------------------
        # Recorder instance
        # --------------------------
        recorder = RIRRecorder(
            in_stream=self.engine,
            in_rate=self.engine.rate,
            in_channels=self.engine.in_channels,
            output_device_index=output_index,
            # sweep_file='Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav',
            sweep_file=self.sweep_file_path,
            record_file=self.rir_save_path,
            record_length=record_length,
            current_blocksize=self.engine.buffer_size
        )

        recorder.log_callback = lambda msg: self._log(msg)

        self._log("[RIRTab] Starting RIR measurement...")

        # --------------------------
        # Run in background thread
        # --------------------------
        threading.Thread(target=recorder.measure, daemon=True).start()


    def choose_sweep_file(self):
        path = filedialog.askopenfilename(
            title="Select sweep file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            self.sweep_file_path = path
            self.sweep_file_label_var.set(os.path.basename(path))

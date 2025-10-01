import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

from utils.rir_record import RIRRecorder


class RIRTab:
    def __init__(self, notebook, engine):
        self.engine = engine
        self.frame = ttk.Frame(notebook)

        # ------------------------------------------------------------------
        # Save path selection
        # ------------------------------------------------------------------
        ttk.Label(self.frame, text="Save RIR As:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.rir_save_label_var = tk.StringVar(value="(none)")
        ttk.Label(self.frame, textvariable=self.rir_save_label_var).grid(
            row=0, column=1, sticky="w", padx=5, pady=5
        )
        ttk.Button(self.frame, text="Browse...", command=self.choose_save_file).grid(
            row=0, column=2, padx=5, pady=5
        )

        # ------------------------------------------------------------------
        # Record length input
        # ------------------------------------------------------------------
        ttk.Label(self.frame, text="Record Length (sec):").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        self.record_length_var = tk.StringVar(value="20")
        ttk.Entry(self.frame, textvariable=self.record_length_var, width=5).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )

        # ------------------------------------------------------------------
        # Measure button
        # ------------------------------------------------------------------
        self.rir_btn = ttk.Button(self.frame, text="Measure RIR", command=self.start_rir_measurement)
        self.rir_btn.grid(row=1, column=2, padx=5, pady=5)

        # ------------------------------------------------------------------
        # Log box
        # ------------------------------------------------------------------
        ttk.Label(self.frame, text="Log:").grid(row=2, column=0, sticky="ne", padx=5, pady=5)
        self.log_box = tk.Text(self.frame, height=12, width=70)
        self.log_box.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

    # ----------------------------------------------------------------------
    # Logging helper
    # ----------------------------------------------------------------------
    def log(self, msg: str):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

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
        try:
            record_length = int(self.record_length_var.get())
        except ValueError:
            messagebox.showwarning("Invalid record length", "Using default of 20 sec.")
            record_length = 20

        # Preconditions
        if self.engine.in_stream is None:
            self.log("[RIRTab] Audio stream not running. Start streaming first.")
            return
        if not hasattr(self, "rir_save_path") or not self.rir_save_path:
            messagebox.showwarning("Save path", "Select a file path to save the measured RIR first.")
            return

        # Recorder instance
        recorder = RIRRecorder(
            in_stream=self.engine,
            in_rate=self.engine.in_rate,
            in_channels=self.engine.in_channels,
            output_device_index=self.engine.output_device_index,
            sweep_file='Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav',
            record_file=self.rir_save_path,
            record_length=record_length,
            current_blocksize=self.engine.buffer_size
        )

        # Connect log callback
        recorder.log_callback = lambda msg: self.log(msg)

        self.log("[RIRTab] Starting RIR measurement...")

        # Run in background
        threading.Thread(target=recorder.measure, daemon=True).start()

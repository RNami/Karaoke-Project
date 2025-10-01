"""
audio_app.py
Tkinter GUI for controlling the Karaoke Reverb audio engine.
"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from Core.audio_engine import AudioEngine
from Core.io_utils import get_wasapi_devices
from utils.rir_record import RIRRecorder


class AudioApp:
    """
    Main GUI application for controlling the audio engine.

    Features:
        - Select input and output devices.
        - Choose audio effects (robot voice, concert hall, convolver).
        - Load impulse response (IR) files for convolution.
        - Display microphone level and current pitch.
        - Start/stop the audio processing stream.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Karaoke Reverb GUI (Two-Stream)")

        # Audio engine instance
        self.engine = AudioEngine()

        # ---------------------------------------------------------------------
        # Device selection
        # ---------------------------------------------------------------------
        self.devices = get_wasapi_devices(self.engine.pa)
        self.input_devices = [d for d in self.devices if d["max_in"] > 0]
        self.output_devices = [d for d in self.devices if d["max_out"] > 0]

        ttk.Label(root, text="Input Device:").grid(row=0, column=0, sticky="e")
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(
            root, textvariable=self.in_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.input_devices],
            state="readonly", width=50
        )
        self.in_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(root, text="Output Device:").grid(row=1, column=0, sticky="e")
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(
            root, textvariable=self.out_var,
            values=[f"[{d['index']}] {d['name']}" for d in self.output_devices],
            state="readonly", width=50
        )
        self.out_combo.grid(row=1, column=1, padx=5, pady=5)

        # ---------------------------------------------------------------------
        # Effect selection
        # ---------------------------------------------------------------------
        ttk.Label(root, text="Effect:").grid(row=2, column=0, sticky="e")
        self.effect_var = tk.StringVar(value="None")
        self.effect_combo = ttk.Combobox(
            root, textvariable=self.effect_var,
            values=["None", "Robot Voice", "Concert Hall", "Convolver"],
            state="readonly"
        )
        self.effect_combo.grid(row=2, column=1, padx=5, pady=5)
        self.effect_combo.bind("<<ComboboxSelected>>", self.on_effect_changed)

        # ---------------------------------------------------------------------
        # Impulse Response (IR) selection
        # ---------------------------------------------------------------------
        ttk.Label(root, text="IR File:").grid(row=3, column=0, sticky="e")
        self.ir_path_var = tk.StringVar(value="(none)")
        self.ir_label = ttk.Label(
            root, textvariable=self.ir_path_var, width=50, anchor="w"
        )
        self.ir_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        self.ir_btn = ttk.Button(root, text="Choose IR...", command=self.choose_ir)
        self.ir_btn.grid(row=3, column=2, padx=5, pady=5)

        # ---------------------------------------------------------------------
        # Mic level meter
        # ---------------------------------------------------------------------
        ttk.Label(root, text="Mic Level:").grid(row=4, column=0, sticky="e")
        self.level = tk.DoubleVar()
        self.level_bar = ttk.Progressbar(
            root, variable=self.level, maximum=100, length=300
        )
        self.level_bar.grid(row=4, column=1, padx=5, pady=5)

        # ---------------------------------------------------------------------
        # Pitch detection display
        # ---------------------------------------------------------------------
        self.note = tk.StringVar()
        ttk.Label(root, text="Current Pitch:").grid(row=5, column=0, sticky="e")
        ttk.Label(root, textvariable=self.note).grid(row=5, column=1, sticky="w")

        # ---------------------------------------------------------------------
        # Start/Stop controls
        # ---------------------------------------------------------------------
        self.start_btn = ttk.Button(root, text="Start", command=self.start_audio)
        self.start_btn.grid(row=6, column=0, padx=5, pady=10)

        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop_audio, state="disabled")
        self.stop_btn.grid(row=6, column=1, padx=5, pady=10)

        # ---------------------------------------------------------------------
        # RIR Measurement Button
        # ---------------------------------------------------------------------
        # --- Save path selection (row 7) ---
        ttk.Label(root, text="Save RIR As:").grid(row=7, column=0, sticky="e")
        self.rir_save_label_var = tk.StringVar(value="(none)")
        ttk.Label(root, textvariable=self.rir_save_label_var).grid(row=7, column=1, sticky="w", padx=5, pady=5)
        ttk.Button(root, text="Browse...", command=self.choose_save_file).grid(row=7, column=2, padx=5, pady=5)

        # Record length input
        ttk.Label(root, text="Record Length (sec):").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        self.record_length_var = tk.StringVar(value="20")  # default to 20 sec
        ttk.Entry(root, textvariable=self.record_length_var, width=5).grid(row=8, column=1, sticky="w", padx=5, pady=5)

        # Measure RIR button
        self.rir_btn = ttk.Button(root, text="Measure RIR", command=self.start_rir_measurement)
        self.rir_btn.grid(row=8, column=2, padx=5, pady=5)

        # --- Log box (row 9) ---
        self.log_box = tk.Text(root, height=6, width=70)
        self.log_box.grid(row=9, column=0, columnspan=3, padx=5, pady=5)

    def log(self, msg: str):
        """Append a message to the log box."""
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------
    def choose_ir(self):
        """Prompt user to choose an IR file and load it if possible."""
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
        """If 'Convolver' is selected without an IR, prompt user to choose one."""
        eff = self.effect_var.get()
        if eff == "Convolver" and (self.engine.convolver is None and not self.engine.ir_path):
            if messagebox.askyesno("Convolver selected", "No IR loaded. Do you want to choose an IR file now?"):
                self.choose_ir()

    def start_audio(self):
        """Start the audio engine with selected devices and effect."""
        try:
            i_idx, o_idx = self.in_combo.current(), self.out_combo.current()
            if i_idx == -1 or o_idx == -1:
                messagebox.showerror("Error", "Select both input and output devices.")
                return

            inp, out = self.input_devices[i_idx], self.output_devices[o_idx]
            effect = self.effect_var.get()

            # Start the audio stream
            self.engine.start_stream(inp["index"], out["index"], effect)

            # Load IR if needed
            if effect == "Convolver" and self.engine.convolver is None and self.engine.ir_path:
                try:
                    self.engine.load_ir(self.engine.ir_path, target_fs=self.engine.in_rate)
                except Exception as e:
                    messagebox.showwarning("IR warning", f"Failed to initialize convolver IR:\n{e}")

            # Update GUI state
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")

            self.update_level_bar()
            self.update_pitch_label()

            threading.Thread(target=self.monitor_stream, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Could not start audio:\n{e}")

    def stop_audio(self):
        """Stop the audio engine and reset GUI buttons."""
        try:
            self.engine.stop_stream()
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Could not stop audio:\n{e}")

    def monitor_stream(self):
        """Background thread that watches the audio stream state and updates GUI."""
        while self.engine.running:
            if not self.engine.in_stream or not self.engine.in_stream.is_active():
                break
            time.sleep(0.05)

        self.root.after(0, lambda: (
            self.start_btn.config(state="normal"),
            self.stop_btn.config(state="disabled")
        ))

    # -------------------------------------------------------------------------
    # GUI update loops
    # -------------------------------------------------------------------------
    def update_level_bar(self):
        """Refresh microphone level bar periodically."""
        self.level.set(self.engine.current_level)
        if self.engine.running:
            self.root.after(50, self.update_level_bar)

    def update_pitch_label(self):
        """Refresh detected pitch display periodically."""
        self.note.set(self.engine.current_note)
        if self.engine.running:
            self.root.after(50, self.update_pitch_label)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def on_close(self):
        """Gracefully stop engine and close the GUI window."""
        self.engine.running = False
        self.engine.stop_stream()
        self.engine.terminate()
        self.root.destroy()


    def start_rir_measurement(self):

        try:
            record_length = int(self.record_length_var.get())
        except ValueError:
            messagebox.showwarning("Invalid record length, using default 20 sec.")
            return

        if self.engine.in_stream is None:
            print("[AudioApp] Audio stream not running.")
            return
        if not hasattr(self, "rir_save_path") or not self.rir_save_path:
            messagebox.showwarning("Save path", "Select a file path to save the measured RIR first.")
            return

        recorder = RIRRecorder(
            in_stream=self.engine,
            in_rate=self.engine.in_rate,
            in_channels=self.engine.in_channels,
            output_device_index=self.engine.output_device_index,  # <<< add this
            sweep_file='Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav',
            record_file=self.rir_save_path,
            record_length=record_length,
            current_blocksize=self.engine.buffer_size
        )

        recorder.log_callback = lambda msg: (
            self.log_box.insert(tk.END, msg + "\n") or self.log_box.see(tk.END)
        )

        self.log("[AudioApp] Starting RIR measurement...")

        threading.Thread(target=recorder.measure, daemon=True).start()



    def choose_save_file(self):
        """Prompt user to select a path to save the measured RIR WAV file."""
        path = filedialog.asksaveasfilename(
            title="Save measured RIR as...",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if path:
            self.rir_save_path = path
            self.rir_save_label_var.set(os.path.basename(path))



if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

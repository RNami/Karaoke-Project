# ui_realtime.py
import os
import threading
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import soundfile as sf
import pyaudio

# Import your existing helpers
from realtime_conv import run_realtime_convolution, load_ir_any
from in_out import BLOCK  # default block size

# ---------- Config ----------
DEFAULT_IR_DIR = r"E:\Uni Lessones\FAU\Semester 2\Summer School\karaoke Project\IR Database\AIR_1_4"

# ---------- Device listing (local — no prompt) ----------
def list_devices():
    p = pyaudio.PyAudio()
    inputs, outputs = [], []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", f"Device {i}")
        max_in  = int(info.get("maxInputChannels", 0))
        max_out = int(info.get("maxOutputChannels", 0))
        fs      = int(info.get("defaultSampleRate", 0) or 0)
        if max_in  > 0:
            inputs.append((i, name, max_in, fs))
        if max_out > 0:
            outputs.append((i, name, max_out, fs))
    p.terminate()
    return inputs, outputs

# ---------- IR normalization helper ----------
def normalize_ir_to_temp(ir_path: str) -> str:
    """
    Load IR (.mat/.wav) via your realtime_conv loader, normalize peak to 1,
    write to a temp .wav, and return the new path.
    """
    ir, fs_ir = load_ir_any(ir_path)  # returns float32, shape (M, C)
    peak = float(np.max(np.abs(ir))) if ir.size else 0.0
    if peak > 1e-12:
        ir = ir / peak
    # Save to temp wav in system temp dir
    tmpdir = tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(ir_path))[0]
    out_path = os.path.join(tmpdir, f"{base}_norm.wav")
    sf.write(out_path, ir.astype(np.float32), fs_ir)
    return out_path

# ---------- Worker launcher (thread) ----------
def launch_realtime(ir_path, block, in_idx, out_idx, fs, wet, dry, normalize):
    try:
        run_path = ir_path
        if normalize:
            run_path = normalize_ir_to_temp(ir_path)
        run_realtime_convolution(
            ir_path=run_path,
            block=block,
            in_device_index=in_idx if in_idx != "" else None,
            out_device_index=out_idx if out_idx != "" else None,
            sample_rate=int(fs) if fs else None,
            wet=wet,
            dry=dry,
        )
    except Exception as e:
        messagebox.showerror("Runtime error", str(e))

# ---------- GUI ----------
class UI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Convolution – Device & IR Selector")
        self.geometry("760x520")

        # State
        self.ir_dir_var = tk.StringVar(value=DEFAULT_IR_DIR)
        self.ir_path_var = tk.StringVar(value="")
        self.normalize_var = tk.BooleanVar(value=True)
        self.block_var = tk.StringVar(value=str(BLOCK))  # samples
        self.fs_var = tk.StringVar(value="")             # blank = device default
        self.wet_var = tk.DoubleVar(value=1.0)
        self.dry_var = tk.DoubleVar(value=0.0)

        self.inputs, self.outputs = list_devices()
        self.in_choice = tk.StringVar()
        self.out_choice = tk.StringVar()

        # Layout
        self._build()

        # Populate device dropdowns
        self._populate_devices()

        # Initial IR scan
        self._scan_ir_folder()

    # --- UI building ---
    def _build(self):
        pad = {"padx": 8, "pady": 6}

        # IR folder & list
        frm_ir = ttk.LabelFrame(self, text="Impulse Response")
        frm_ir.pack(fill="x", **pad)

        row1 = ttk.Frame(frm_ir); row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Folder:").pack(side="left")
        ttk.Entry(row1, textvariable=self.ir_dir_var, width=90).pack(side="left", padx=6)
        ttk.Button(row1, text="Browse…", command=self._browse_ir_dir).pack(side="left")
        ttk.Button(row1, text="Rescan", command=self._scan_ir_folder).pack(side="left")

        row2 = ttk.Frame(frm_ir); row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Files:").pack(side="left")
        self.ir_list = tk.Listbox(row2, height=8, width=100)
        self.ir_list.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(row2, orient="vertical", command=self.ir_list.yview)
        sb.pack(side="left", fill="y")
        self.ir_list.configure(yscrollcommand=sb.set)

        row3 = ttk.Frame(frm_ir); row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Selected IR:").pack(side="left")
        ttk.Entry(row3, textvariable=self.ir_path_var, width=100).pack(side="left", padx=6)
        ttk.Checkbutton(row3, text="Normalize IR (peak = 1.0)", variable=self.normalize_var).pack(side="left", padx=12)

        # Devices
        frm_dev = ttk.LabelFrame(self, text="Devices")
        frm_dev.pack(fill="x", **pad)

        row4 = ttk.Frame(frm_dev); row4.pack(fill="x", **pad)
        ttk.Label(row4, text="Input: ").pack(side="left")
        self.in_combo = ttk.Combobox(row4, state="readonly", width=80, textvariable=self.in_choice)
        self.in_combo.pack(side="left", padx=6)

        row5 = ttk.Frame(frm_dev); row5.pack(fill="x", **pad)
        ttk.Label(row5, text="Output:").pack(side="left")
        self.out_combo = ttk.Combobox(row5, state="readonly", width=80, textvariable=self.out_choice)
        self.out_combo.pack(side="left", padx=6)

        row6 = ttk.Frame(frm_dev); row6.pack(fill="x", **pad)
        ttk.Label(row6, text="Sample rate (Hz, blank=default):").pack(side="left")
        ttk.Entry(row6, textvariable=self.fs_var, width=12).pack(side="left", padx=6)

        # DSP settings
        frm_cfg = ttk.LabelFrame(self, text="Processing")
        frm_cfg.pack(fill="x", **pad)

        row7 = ttk.Frame(frm_cfg); row7.pack(fill="x", **pad)
        ttk.Label(row7, text="Block size (samples):").pack(side="left")
        ttk.Entry(row7, textvariable=self.block_var, width=8).pack(side="left", padx=6)
        ttk.Label(row7, text="Wet").pack(side="left", padx=(16, 0))
        ttk.Scale(row7, variable=self.wet_var, from_=0.0, to=1.0, length=200).pack(side="left")
        ttk.Label(row7, text="Dry").pack(side="left", padx=(16, 0))
        ttk.Scale(row7, variable=self.dry_var, from_=0.0, to=1.0, length=200).pack(side="left")

        # Run
        frm_run = ttk.Frame(self); frm_run.pack(fill="x", **pad)
        ttk.Button(frm_run, text="Start real-time convolution", command=self._on_start).pack(side="left")
        ttk.Label(frm_run, text="(Close console/press Ctrl+C to stop stream)").pack(side="left", padx=12)

    # --- Handlers ---
    def _browse_ir_dir(self):
        d = filedialog.askdirectory(initialdir=self.ir_dir_var.get() or DEFAULT_IR_DIR)
        if d:
            self.ir_dir_var.set(d)
            self._scan_ir_folder()

    def _scan_ir_folder(self):
        self.ir_list.delete(0, tk.END)
        folder = self.ir_dir_var.get()
        if not folder or not os.path.isdir(folder):
            return
        # List .mat and .wav in the chosen directory (non-recursive for speed)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".mat", ".wav"))]
        files.sort()
        for f in files:
            self.ir_list.insert(tk.END, f)
        # Click loads into Selected IR
        def on_select(evt):
            sel = self.ir_list.curselection()
            if sel:
                fname = self.ir_list.get(sel[0])
                self.ir_path_var.set(os.path.join(folder, fname))
        self.ir_list.bind("<<ListboxSelect>>", on_select)

    def _populate_devices(self):
        # Build pretty labels and keep mapping index->id
        in_labels = [f"[{i}] {name}  ch:{ch}  fs:{fs}" for (i, name, ch, fs) in self.inputs]
        out_labels = [f"[{i}] {name}  ch:{ch}  fs:{fs}" for (i, name, ch, fs) in self.outputs]
        self.in_combo["values"] = in_labels
        self.out_combo["values"] = out_labels
        if in_labels:
            self.in_combo.current(0)
        if out_labels:
            self.out_combo.current(0)

    def _parse_selected_device(self, combo, options):
        idx = combo.current()
        if idx < 0:
            return None
        dev_id = options[idx][0]  # PyAudio device index
        return dev_id

    def _on_start(self):
        ir_path = self.ir_path_var.get().strip()
        if not ir_path or not os.path.exists(ir_path):
            messagebox.showerror("Select IR", "Please select an impulse response file (.mat or .wav).")
            return

        # Parse devices
        in_dev = self._parse_selected_device(self.in_combo, self.inputs)
        out_dev = self._parse_selected_device(self.out_combo, self.outputs)

        # Parse integers
        try:
            block = int(self.block_var.get())
            if block <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Block size", "Block size must be a positive integer (e.g., 256 or 1024).")
            return

        fs_str = self.fs_var.get().strip()
        fs = int(fs_str) if fs_str else None

        wet = float(self.wet_var.get())
        dry = float(self.dry_var.get())
        normalize = bool(self.normalize_var.get())

        # Start the processing in a background thread (so UI stays responsive)
        t = threading.Thread(
            target=launch_realtime,
            args=(ir_path, block, in_dev, out_dev, fs, wet, dry, normalize),
            daemon=True
        )
        t.start()
        messagebox.showinfo("Running", "Streaming started.\n\nUse the console window to stop (Ctrl+C).")

if __name__ == "__main__":
    app = UI()
    app.mainloop()

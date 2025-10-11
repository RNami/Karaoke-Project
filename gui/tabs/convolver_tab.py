import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import sounddevice as sd
import threading

from scipy.signal import fftconvolve

from gui.base_widgets import (
    make_label,
    make_button,
    make_combobox,
    make_progressbar,
    make_frame,
    configure_grid
)

class ConvolverTab:
    def __init__(self, notebook, engine):
        self.engine = engine
        self.frame = make_frame(notebook)
        configure_grid(self.frame, cols=4, rows=4, weight=1)

        self.input_audio = None # Loaded audio (float32)
        self.sample_rate = None
        self.ir_convolver = None
        self.output_audio = None


        # ----------------------------
        # Widgets
        # ----------------------------
        make_label(self.frame, text="Audio File:",
                   row=0, column=0)
        self.audio_label = make_label(self.frame, text="None",
                                      row=0, column=1, sticky="w")
        make_button(self.frame, "Load Audio...",
                    command=self.load_audio,
                    row=0, column=2)
        
        make_label(self.frame, text="Impulse Response (IR):",
                   row=1, column=0)
        self.ir_label = make_label(self.frame, text="None",
                                   row=1, column=1, sticky="w")
        make_button(self.frame, "Load IR...",
                    command=self.load_ir,
                    row=1, column=2)
        make_button(self.frame, "Convolve",
                    command=self.convolve_audio_fft,
                    row=2, column=0,
                    columnspan=3, sticky="we")
        make_button(self.frame, "Play Output",
                    command=self.play_output,
                    row=3, column=0,
                    columnspan=3, sticky="we")
        make_button(self.frame, "Save Output",
                    command=self.save_output,
                    row=4, column=0,
                    columnspan=3, sticky="we")
        

    # ----------------------------
    # Load audio
    # ----------------------------
    def load_audio(self):
        file = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.ogg *.mp3")]
        )
        if not file:
            return
        try:
            data, sr = sf.read(file, dtype='float32')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio:\n{e}")
            return
        self.input_audio = data
        self.sample_rate = sr
        self.audio_label.config(text=file.split('/')[-1])
        messagebox.showinfo("Audio Loaded", f"Audio loaded: {data.shape} samples at {sr} Hz")

    # ----------------------------
    # Load Impulse Response
    # ----------------------------
    def load_ir(self):
        file = filedialog.askopenfilename(
            title="Select IR File",
            filetypes=[("All files", "*.wav *.ogg *.mat *.npy"), ("Audio files", "*.wav *.ogg"), ("Other files", "*.mat *.npy")]
        )
        if not file:
            return
        try:
            if file.endswith(".npy"):
                ir = np.load(file)
                sr = self.sample_rate if self.sample_rate else 44100
            else:
                ir, sr = sf.read(file, dtype='float32')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load IR:\n{e}")
            return
        
        self.ir = ir
        self.ir_sr = sr
        self.ir_label.config(text=file.split('/')[-1])
        messagebox.showinfo("IR loaded", f"IR loaded: {ir.shape} samples at {sr} Hz")

    # ----------------------------
    # Convolve (FFT)
    # ----------------------------
    def convolve_audio_fft(self) -> np.ndarray: # supporting stereo in/out
        """
        Convolve input_audio with IR using FFT-based convolution.

        input_audio: shape (L, in_ch)
        ir: shape (M, out_ch)

        Returns: shape (L + M - 1, out_ch)
        """
        if self.input_audio is None or self.ir is None:
            messagebox.showwarning("Missing Data", "Please load both an audio file and an IR.")
            return
        
        x = self.input_audio
        h = self.ir

        # Ensure 2D arrays for both
        if x.ndim == 1:
            x = x[:, None]
        if h.ndim == 1:
            h = h[:, None]

        # if input_audio.ndim == 1:
        #     input_audio = input_audio[:, None] # make mono input (L, 1)
        # if ir.ndim == 1:
        #     ir = ir[:, None] # make mono IR (M, 1)

        L, in_ch = x.shape
        M, out_ch = h.shape

        output = np.zeros((L + M - 1), out_ch, dtype=np.float32)

        # convolve each IR output channel with all input channels, sum contributions
        for c_out in range(out_ch):
            conv_sum = np.zeros(L + M - 1, dtype=np.float32)
            for c_in in range(in_ch):
                conv_sum += fftconvolve(input_audio[:, c_in], ir[:, c_out], mode='full')
            output[:, c_out] = conv_sum

        # Normalize output if needed to prevent clipping
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        self.output_audio = output
        messagebox.showinfo("Done", f"Convolution complete! Output shape {output.shape}")

        return output

    # ----------------------------
    # Playback (sounddevice)
    # ----------------------------
    def play_output(self):
        if self.output_audio is None:
            messagebox.showwarning("No Output", "Please convolve first.")
            return
        
        def _play():
            try:
                sd.play(self.output_audio, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Error", f"Playback failed:\n{e}")

        threading.Thread(target=_play, daemon=True).start()
        

    def save_output(self):
        pass
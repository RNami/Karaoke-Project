import tkinter as tk
from tkinter import ttk

class AboutTab:
    def __init__(self, notebook):
        self.frame = ttk.Frame(notebook)
        text = tk.Text(self.frame, wrap="word", width=60, height=15)
        text.insert(tk.END, 
            "Karaoke Reverb 1.0\n"
            "1 Oct 2025\n"
            "Part of FERIENAKADEMIE 2025 Course 10\n\n"
            "Developers:\n"
            "Samuel, Philip, Romina, Jeongjoo, Bhavya, Khashayar, Shitao\n\n"
            "made with <3 in Sarntal"
        )
        text.config(state="disabled")
        text.pack(expand=True, fill="both")

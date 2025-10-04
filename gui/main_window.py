import tkinter as tk
from tkinter import ttk

from Core.audio_engine import AudioEngine
from Core.io_utils import get_wasapi_devices
from gui.tabs.streaming_tab import StreamingTab
from gui.tabs.rir_tab import RIRTab
from gui.tabs.about_tab import AboutTab

class AudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Karaoke Reverb GUI")

        # ----------------------------
        # Shared audio engine instance
        # ----------------------------
        self.engine = AudioEngine(buffer_size=128)
        
        # ----------------------------
        # Notebook for tabs
        # ----------------------------
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)

        self.streaming_tab = StreamingTab(notebook, self.engine, get_wasapi_devices(self.engine.pa))
        self.rir_tab = RIRTab(notebook, self.engine, get_wasapi_devices(self.engine.pa))
        self.about_tab = AboutTab(notebook)

        notebook.add(self.streaming_tab.frame, text="Streaming")
        notebook.add(self.rir_tab.frame, text="RIR Measurement")
        notebook.add(self.about_tab.frame, text="About")

        # ----------------------------
        # Handle closing the app
        # ----------------------------
        root.protocol("WM_DELETE_WINDOW", self.on_close)


    def on_close(self):
        self.engine.stop_stream()
        self.engine.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

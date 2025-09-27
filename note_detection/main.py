"""
main.py
Entry point for the project.
"""

from note_detection.audio_app import AudioApp
import tkinter as tk

def main():
    """Main function controlling the application flow."""
    root = tk.Tk()
    app = AudioApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

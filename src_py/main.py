import sys
try:
    import tkinter as tk
except ImportError:
    print("Error: Tkinter is missing.")
    print("Please install it using your system package manager:")
    print("  Ubuntu/Debian: sudo apt install python3-tk")
    print("  Fedora:        sudo dnf install python3-tkinter")
    print("  Arch:          sudo pacman -S tk")
    sys.exit(1)
from gui import AudioStreamerGUI

def main():
    root = tk.Tk()
    app = AudioStreamerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

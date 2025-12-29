import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import sys
from collections import deque

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.style as mpl_style

from wrapper import AudioEngineWrapper

mpl_style.use('dark_background')

class AudioStreamerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Audio Streamer")
        self.root.geometry("600x700")

        # --- Data Buffer for Plotting
        # Stores the last 60 points. i.e., if poll 100ms, shows 6 seconds of history.
        self.perf_data = deque([0]*60, maxlen=60)
        self.current_tick = 60

        # Build the UI
        self._create_widgets()

        # Redirect prints
        sys.stdout = PrintRedirector(self.log_to_gui)
        sys.stderr = PrintRedirector(self.log_to_gui)

        # Initialize the C Audio Engine
        try:
            self.engine = AudioEngineWrapper()
            self.engine.register_logger(self.log_to_gui)
            self.engine.initialize()
            self.is_running = False
        except Exception as e:
            self.log_to_gui(f"CRITICAL ERROR: {e}\nExiting...\n")
            messagebox.showerror("Error", f"Failed to load Audio Engine:\n{e}")
            self.root.destroy()
            return
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    def _create_widgets(self):
        # --- Header ---
        lbl_title = tk.Label(self.root, text="Audio Streamer Engine", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=20)

        # --- Controls Frame ---
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Load IR Button
        self.btn_load = tk.Button(control_frame, text="Load Impulse Response (WAV)", command=self.load_ir)
        self.btn_load.pack(fill=tk.X, pady=5)

        # Status Label
        self.lbl_status = tk.Label(control_frame, text="Status: Ready (Dry Signal)", fg="gray")
        self.lbl_status.pack(pady=5)

        # Separator
        tk.Frame(self.root, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=20, pady=10)

        # --- Stream Controls ---
        # a single toggle button for better UX
        self.btn_toggle = tk.Button(self.root, text="Start Stream", bg="#dddddd", height=2, command=self.toggle_stream)
        self.btn_toggle.pack(fill=tk.X, padx=40, pady=10)

        # Mix Slider (Placeholder for future implementation)
        lbl_mix = tk.Label(self.root, text="Dry / Wet Mix:")
        lbl_mix.pack()
        self.slider_mix = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_mix)
        self.slider_mix.set(0)
        self.slider_mix.pack(fill=tk.X, padx=40)
        
        # --- Performance Monitor ---
        # self.lbl_perf = tk.Label(self.root, text="CPU Load: 0.0%", font=("Consolas", 10))
        # self.lbl_perf.pack(pady=5)

        # self.lbl_perf = tk.Label(self.root, text="CPU Load (%)")
        self.lbl_perf = tk.Label(self.root, text="CPU Load (%)", font=("Consolas", 10))
        self.lbl_perf.pack(pady=(10, 0))

        self.fig = Figure(figsize=(5, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("CPU Load (%)")
        self.ax.set_xticklabels([])
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False)
        self.ax.grid(True, color="#333333", linestyle='-', alpha=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        # --- Logs ---
        lbl_logs = tk.Label(self.root, text="Debug Logs", anchor="w")
        lbl_logs.pack(fill=tk.X, padx=10, pady=(20, 0))

        self.txt_logs = scrolledtext.ScrolledText(self.root, height=10, state='disabled', bg="black", fg="#00ff00", font=("Consolas", 9))
        self.txt_logs.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def update_performance_stats(self):
        """Fetches stats from C every 100ms"""
        if self.is_running:
            if hasattr(self.engine, 'get_performance'):
                load = self.engine.get_performance()
            else:
                load = self.engine.get_performance()

            self.perf_data.append(load)
            self.current_tick += 1

            self.ax.clear()
            x_start = self.current_tick - 60
            x_end = self.current_tick

            self.ax.set_xlim(x_start, x_end)
            self.ax.set_ylim(0, 100)
            self.ax.grid(True, color="#333333", linestyle='-', alpha=0.3)
            self.ax.set_xticklabels([])
            self.ax.tick_params(axis='x', which='both', bottom=False, top=False)
            self.ax.set_ylabel("CPU Load (%)")

            x_values = range(x_start, x_end)
            y_values = list(self.perf_data)

            # Color code Green < 50%, Orange < 90%, Red > 90%
            color = "#00ff00"
            if load > 50: color = "#ffa500"
            if load > 90: color = "#ff0000"

            self.ax.plot(x_values, y_values, color=color, linewidth=1.5)
            self.ax.fill_between(x_values, y_values, color=color, alpha=0.2)
            self.lbl_perf.config(text=f"CPU Load: {load:.2f}%", fg=color)
            self.canvas.draw_idle()
        
        # Schedule next check in 500 ms
        self.root.after(100, self.update_performance_stats)

    def log_to_gui(self, message):
        """
        Inserts message into the text box.
        Thread-safe note: If C calls this from a different thread, we might need
        root.after(), but for now, Miniaudio init/start usually happens on main thread.
        """
        # Enable editing just to write
        self.txt_logs.config(state='normal')

        # Define tags for coloring if they don't already exist
        self.txt_logs.tag_config("c_tag", foreground="cyan")
        self.txt_logs.tag_config("py_tag", foreground="yellow")
        self.txt_logs.tag_config("err_tag", foreground="red")

        tag_to_use = None
        if "[C Engine]" in message:
            tag_to_use = "c_tag"
        elif "[Python Wrapper]" in message:
            tag_to_use = "py_tag"
        elif "[ERROR]" in message:
            tag_to_use = "err_tag"

        # Insert text at the end
        if not message.endswith('\n'):
            message += '\n'

        self.txt_logs.insert(tk.END, message, tag_to_use)

        # Auto-scroll to bottom
        self.txt_logs.see(tk.END)
        # Disable editing again (read-only)
        self.txt_logs.config(state='disabled')

        # Force GUI update
        self.root.update_idletasks()
    
    def on_close(self):
        """Safely shuts down the C engine before closing the window."""
        if self.is_running: self.engine.stop()
        self.engine.close()
        self.root.destroy()
        sys.exit(0)

    def load_ir(self):
        filename = filedialog.askopenfilename(
            title="Select Impulse Response",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )

        if filename:
            self.engine.load_ir_file(filename)
            # Update GUI to show file loaded
            short_name = filename.split("/")[-1]
            self.lbl_status.config(text=f"IR Loaded: {short_name}", fg="green")

    def toggle_stream(self):
        if not self.is_running:
            # Start
            self.engine.start()
            self.btn_toggle.config(text="Stop Stream", bg="#ffcccc")
            self.is_running = True

            self.update_performance_stats()
        else:
            # Stop
            self.engine.stop()
            self.btn_toggle.config(text="Start Stream", bg="#dddddd")
            self.is_running = False

    def update_mix(self, value):
        # Convert 0-100 scale to 0.0-1.0 float
        float_val = float(value) / 100.0
        self.engine.set_mix(float_val)


# Helper class to catch Python "print" statements
class PrintRedirector:
    def __init__(self, func):
        self.func = func
    def write(self, string):
        if string.strip() != "": self.func(string)
    def flush(self): pass
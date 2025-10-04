# Karaoke Reverb & Real-Time Audio Engine

A **real-time audio processing and karaoke project** with effects, impulse response (IR) convolution, and RIR (Room Impulse Response) measurement, built in **Python** using **Tkinter** for GUI and **PyAudio** for audio streaming.

---

## Features

### Real-Time Streaming
- Capture audio from microphone and playback with low latency.
- Apply effects:
  - None
  - Robot Voice
  - Concert Hall (custom reverb)
  - Convolver (load your own IR)
- Visualize:
  - Mic level progress bar
  - Current pitch detection (monophonic)
- Start/Stop streaming with responsive UI buttons.

### RIR Measurement
- Measure **Room Impulse Response** using a sine sweep.
- Select **input/output audio devices**.
- Specify recording length and save path.
- Supports browsing for custom sweep audio files.
- Logs measurement progress in a non-editable textbox.

### Convolution / IR
- Load **Impulse Response files** (`.wav`, `.flac`, `.aiff`, `.mat`) for convolver effect.
- Apply IR in real-time to microphone input.
- Adjustable wet/dry mix.

---

## Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd karaoke-reverb
````

2. Create and activate a Python virtual environment:

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Dependencies include `numpy`, `scipy`, `pyaudio`, and any other required packages.

---

## Running the Application

```bash
python -m launcher
```

* Opens the **Tkinter GUI** with tabs:

  * **Streaming**: real-time audio and effects
  * **RIR Measurement**: record room impulse responses

---

## Usage

### Streaming Tab

1. Select **input and output devices**.
2. Choose an **effect**.
3. If using **Convolver**, select an **IR file**.
4. Press **Start** to begin streaming.
5. View mic level and pitch in real-time.
6. Press **Stop** to end streaming.

### RIR Measurement Tab

1. Select **input and output devices**.
2. Choose a **save path** for the RIR.
3. Optionally select a **custom sweep file** (default: `Archive/Sample_Audio/sine-sweep-linear-10sec-48000sr.wav`).
4. Set **record length** in seconds.
5. Press **Measure RIR**.
6. View logs in the non-editable textbox.

---

## File Structure

```
karaoke-reverb/
│
├─ Core/
│  └─ audio_engine.py        # Real-time audio engine 
│
├─ gui/
│  ├─ main_window.py         # Main GUI
│  ├─ tabs/
│  │  ├─ streaming_tab.py    # Streaming tab
│  │  └─ rir_tab.py          # RIR measurement tab
│  └─ base_widgets.py        # Tkinter widget helpers
│
├─ utils/
│  └─ rir/
│     └─ rir_record.py       # RIR measurement logic
│
├─ Archive/
│  └─ Sample_Audio/
│     └─ sine-sweep-linear-10sec-48000sr.wav
|
├─ launcher.py				 # Entry point
│
└─ README.md
```

---

## Notes / Known Issues

* **AudioEngine**:

  * Ensure `running` state is properly set; otherwise UI updates won’t work.
* **StreamingTab**:

  * Uses `after()` loops for mic level and pitch; avoid blocking operations.
* **Device indices**:

  * PyAudio device indices may differ per system.
* **Cross-platform**:

  * Tested on Windows 10/11 with PyAudio; Linux/macOS may require portaudio installation.

---

## Contribution

* Fork the repository, make your changes, and submit a pull request.
* Suggestions for new effects, visualizations, or performance improvements are welcome.


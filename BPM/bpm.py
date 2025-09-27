import numpy as np
import madmom
from collections import deque

# Online beat processor
beat_proc = madmom.features.beats.RNNBeatProcessor()  # neural net that gives beat activation function
tracker = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)  # dynamic Bayesian network tracker

# Keep a sliding buffer of beat activation values
act_buffer = deque(maxlen=1000)  # ~10s if fps=100

def process_audio_block_madmom(x_block, in_rate):
    """Feed one audio block into madmom and estimate BPM in real-time."""

    # Convert int16 -> float32 mono [-1,1]
    if x_block.ndim > 1:
        x_block = x_block.mean(axis=1)
    x_block = x_block.astype(np.float32) / 32768.0

    # Feed into beat processor
    activations = beat_proc(x_block, sr=in_rate)

    # Accumulate beat activations
    act_buffer.extend(activations)

    # Only analyze when buffer is filled enough
    if len(act_buffer) > 200:  # ~2 seconds
        # Convert deque -> numpy
        act_array = np.array(act_buffer)

        # Get beats (timestamps in seconds relative to buffer)
        beats = tracker(act_array)

        if len(beats) > 1:
            # Estimate BPM from inter-beat intervals
            intervals = np.diff(beats)
            bpm = 60.0 / np.median(intervals)
            print(f"[Realtime BPM] {bpm:.1f}")
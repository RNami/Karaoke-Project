"""
AudioInput: convenience wrapper around PyAudio.

Features:
- auto-select an input device by token ("cheap_microphone" or "UA-25EX"),
- open an input stream,
- deliver audio via callback, de-interleaved per channel (one callback per channel),
- allow changing the frame size (frames_per_buffer),
- start/stop safely.

Latency:
    You can call ``test_device_delay()`` to get a rough estimate of input latency.
    This combines:
    - the device's reported input latency (if available),
    - plus the block duration implied by frames_per_buffer.
    Example:

        audio = AudioInput("cheap_microphone", samples_per_frame=1024)
        audio.start_stream()
        print("Latency ~", audio.test_device_delay(), "seconds")

Usage example (see __main__ at the bottom):

    audio = AudioInput(device="cheap_microphone", samples_per_frame=1024)
    chs, sr = audio.get_device_info()
    print(f"Device -> channels={chs}, sample_rate={sr}")

    # one callback per channel
    def cb0(data: np.ndarray) -> None:
        # data.shape == (samples_per_frame,)
        pass

    cbs = [cb0] if chs == 1 else [cb0] * chs
    audio.register_callback(cbs)

    audio.start_stream()
    time.sleep(5)
    audio.stop_stream()
"""

from typing import List, Callable, Any, Optional, Tuple
import inspect
import threading
import time
import queue
import numpy as np
import pyaudio
import contextlib


class AudioInput:
    """
    AudioInput wraps a PyAudio input stream with device auto-selection and
    per-channel callback dispatch.

    Main responsibilities:
    - resolve a device index from a symbolic token ("cheap_microphone" or "UA-25EX"),
    - open/close the corresponding PyAudio stream,
    - convert raw bytes to numpy int16 arrays,
    - split multi-channel data and dispatch each channel to its own callback,
    - optionally buffer the received frames for later retrieval,
    - allow runtime change of frames_per_buffer (frame size),
    - provide a simple latency estimate.

    Typical usage:
        audio = AudioInput("cheap_microphone", 1024)
        chs, sr = audio.get_device_info()
        def cb0(data: np.ndarray): print("Got", data.shape)
        audio.register_callback([cb0])
        audio.start_stream()
        time.sleep(5)
        audio.stop_stream()
    """

    def __init__(self, device: str, samples_per_frame: int) -> None:
        if device not in ["cheap_microphone", "UA-25EX"]:
            raise ValueError(
                f"Unsupported device: {device}. Choose 'cheap_microphone' or 'UA-25EX'."
            )

        self.device_name = device
        self._pa = pyaudio.PyAudio()

        self.device_index, self.device_info = self._resolve_device_by_token(self.device_name)
        if self.device_index is None:
            self._pa.terminate()
            raise ValueError(f"Unable to locate required audio device for '{self.device_name}'.")

        self.input_channels = int(self.device_info.get("maxInputChannels", 0))  # type: ignore
        self.sample_rate = int(self.device_info.get("defaultSampleRate", 0))  # type: ignore
        self.samples_per_frame = int(samples_per_frame)

        self.stream: Optional[pyaudio.Stream] = None
        self._stream_callback = None
        self._callbacks: Optional[List[Callable]] = None
        self._save_to_buffer: bool = False
        self._buffers: Optional[List[List[np.ndarray]]] = None

        self._last_callback_ts: Optional[float] = None
        self._callback_interval_estimate: Optional[float] = None

        self._lock = threading.RLock()

    # -------- Device selection helpers --------
    def _resolve_device_by_token(self, token: str) -> Tuple[Optional[int], Optional[dict]]:
        matched_index = None
        matched_info = None
        for index in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(index)
            name = info.get("name", "")
            if not isinstance(name, str):
                continue

            if token == "cheap_microphone":
                name_lower = name.lower()
                is_usb = "usb" in name_lower
                has_single_input = int(info.get("maxInputChannels", 0)) == 1
                has_no_output = int(info.get("maxOutputChannels", 0)) == 0
                if is_usb and has_single_input and has_no_output:
                    matched_index = index
                    matched_info = info
                    break
            else:  # "UA-25EX"
                if "in (ua-25ex)" in name.lower():
                    matched_index = index
                    matched_info = info
                    break

        return matched_index, matched_info  # type: ignore

    # -------- Stream open/close --------
    def _open_device(self) -> None:
        with self._lock:
            if self.stream is not None:
                return

            self._probe_format_support()

            kwargs = dict(
                format=pyaudio.paInt16,
                channels=self.input_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.samples_per_frame,
            )
            if self._stream_callback is not None:
                kwargs["stream_callback"] = self._stream_callback  # type: ignore

            try:
                self.stream = self._pa.open(**kwargs)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to open audio device: {e}")

    def _close_device(self) -> None:
        with self._lock:
            if self.stream is not None:
                with contextlib.suppress(Exception):
                    if self.stream.is_active():
                        self.stream.stop_stream()
                with contextlib.suppress(Exception):
                    self.stream.close()
                self.stream = None

    def _probe_format_support(self) -> None:
        self._pa.is_format_supported(
            rate=self.sample_rate,
            input_device=self.device_index,
            input_channels=self.input_channels,
            input_format=pyaudio.paInt16,
        )

    # -------- Public API --------
    def get_device_info(self) -> Tuple[int, int]:
        return (self.input_channels, self.sample_rate)

    def _validate_callbacks(self, callbacks: List[Callable]) -> None:
        if not isinstance(callbacks, list) or len(callbacks) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} callback(s), got "
                f"{len(callbacks) if isinstance(callbacks, list) else type(callbacks)}"
            )

        for cb in callbacks:
            if not callable(cb):
                raise TypeError("Idiot! Wrong callback signature!")

            sig = inspect.signature(cb)
            params = [
                p
                for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(params) != 1:
                raise TypeError("Idiot! Wrong callback signature!")

            ann = params[0].annotation
            if ann is not inspect._empty:
                if ann not in (np.ndarray, Any):
                    if not (isinstance(ann, str) and "ndarray" in ann.lower()):
                        raise TypeError("Idiot! Wrong callback signature!")

    def register_callback(self, callbacks: List[Callable], save_to_buffer: bool = False) -> None:
        self._validate_callbacks(callbacks)

        with self._lock:
            self._callbacks = callbacks
            self._save_to_buffer = bool(save_to_buffer)
            self._buffers = [[] for _ in range(self.input_channels)] if self._save_to_buffer else None

            def _stream_callback(in_data, frame_count, time_info, status_flags):
                try:
                    if not in_data:
                        return (None, pyaudio.paContinue)

                    arr = np.frombuffer(in_data, dtype=np.int16)
                    expected = frame_count * self.input_channels
                    if arr.size < expected:
                        return (None, pyaudio.paContinue)
                    if arr.size > expected:
                        arr = arr[:expected]

                    block = arr.reshape(frame_count, self.input_channels)

                    now = time.time()
                    if self._last_callback_ts is not None:
                        self._callback_interval_estimate = now - self._last_callback_ts
                    self._last_callback_ts = now

                    for ch, cb in enumerate(self._callbacks):  # type: ignore
                        ch_data = block[:, ch].copy()
                        cb(ch_data)
                        if self._save_to_buffer:
                            self._buffers[ch].append(ch_data)  # type: ignore

                except Exception:
                    pass

                return (None, pyaudio.paContinue)

            self._stream_callback = _stream_callback

            if self.stream is not None:
                was_active = self.stream.is_active()
                self._close_device()
                self._open_device()
                if was_active and self.stream is not None:
                    with contextlib.suppress(Exception):
                        self.stream.start_stream()

    def reset_frame_size(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Frame size must be > 0.")
        with self._lock:
            self.samples_per_frame = int(size)
            if self.stream is not None:
                was_active = self.stream.is_active()
                self._close_device()
                self._open_device()
                if was_active and self.stream is not None:
                    with contextlib.suppress(Exception):
                        self.stream.start_stream()

    def start_stream(self) -> None:
        with self._lock:
            if self.stream is None:
                self._open_device()
            if not self.stream.is_active():  # type: ignore
                self.stream.start_stream()  # type: ignore

    def stop_stream(self) -> None:
        self._close_device()

    def test_device_delay(self) -> Optional[float]:
        with self._lock:
            if self.stream is None:
                return None

            block_sec = self.samples_per_frame / float(self.sample_rate or 1)
            latency = None
            with contextlib.suppress(Exception):
                latency = self.stream.get_input_latency()
            if latency is None or latency <= 0:
                return block_sec
            return latency + block_sec

    def pop_all_buffers(self) -> Optional[List[np.ndarray]]:
        with self._lock:
            if not self._save_to_buffer or self._buffers is None:
                return None
            out = []
            for ch_list in self._buffers:
                if len(ch_list) == 0:
                    out.append(np.empty((0,), dtype=np.int16))
                else:
                    out.append(np.concatenate(ch_list, axis=0))
            self._buffers = [[] for _ in range(self.input_channels)]
            return out


# ---------- Minimal demo ----------
if __name__ == "__main__":
    import sys, time, queue, math, contextlib
    import numpy as np
    import pyaudio

    token = "cheap_microphone"
    frame = 128

    try:
        audio = AudioInput(device=token, samples_per_frame=frame)
    except Exception as e:
        print(f"Init failed: {e}")
        sys.exit(1)

    chs, sr = audio.get_device_info()
    print(f"[AudioInput] input: channels={chs}, sample_rate={sr}, frames_per_buffer={frame}")

    # ---------- Step 0: Output self-test ----------
    pa = audio._pa
    duration_s = 3.5
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    tone = (0.2 * np.sin(2 * math.pi * 440.0 * t))
    tone_i16 = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)

    try:
        out = pa.open(format=pyaudio.paInt16,
                      channels=2, rate=sr, output=True,
                      frames_per_buffer=frame)
        stereo = np.column_stack([tone_i16, tone_i16]).ravel().tobytes()
        out.write(stereo)
        print("[Diag] 440 Hz tone sent to speakers (stereo).")
    except Exception as e:
        print(f"[Diag] stereo open failed: {e}, trying mono …")
        out = pa.open(format=pyaudio.paInt16,
                      channels=1, rate=sr, output=True,
                      frames_per_buffer=frame)
        out.write(tone_i16.tobytes())
        print("[Diag] 440 Hz tone sent to speakers (mono).")

    # ---------- Step 1: Set up recording callbacks ----------
    q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)

    cb_counter = 0
    last_print = 0.0

    def make_callback(idx: int):
        def audio_input_callback(data: np.ndarray) -> None:
            global cb_counter, last_print
            cb_counter += 1
            now = time.time()
            if now - last_print > 0.1:
                print(f"[CB] ch={idx}, frame_size={data.size}, total_cb={cb_counter}")
                last_print = now
            if idx == 0:
                stereo = np.column_stack([data, data]).ravel().astype(np.int16, copy=False).tobytes()
                try:
                    q.put_nowait(stereo)
                except queue.Full:
                    pass
        return audio_input_callback

    callbacks = [make_callback(i) for i in range(chs)]
    audio.register_callback(callbacks, save_to_buffer=False)

    # ---------- Step 2: Start input; pump queue → output ----------
    audio.start_stream()
    with contextlib.suppress(Exception):
        out.start_stream()

    print("[Loopback] reading mic → playing to default output for 8 s …")
    t0 = time.time()
    try:
        while time.time() - t0 < 8.0:
            try:
                block = q.get(timeout=0.2)
                out.write(block)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass

    # ---------- Step 3: Diagnostics ----------
    est = audio.test_device_delay()
    if est is not None:
        print(f"[Diag] rough input latency estimate: {est*1000:.1f} ms")
    print(f"[Diag] callback count: {cb_counter}")

    # ---------- Step 4: Cleanup ----------
    audio.stop_stream()
    with contextlib.suppress(Exception):
        if out.is_active():
            out.stop_stream()
    with contextlib.suppress(Exception):
        out.close()
    print("[Done] stopped.")

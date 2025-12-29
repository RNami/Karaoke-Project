import ctypes
import os
import sys
import platform

LOG_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p)

class AudioEngineWrapper:
    def __init__(self):
        self._lib = None
        # Keeping a reference to the callback sp garbage collection doesn't kill it
        self._log_bc_ref = None
        self._load_library()
        self._configure_functions()

    def _log(self, message):
        """Internal helper to standardize log formatting."""
        print(f"[Python Wrapper] {message}")

    def _load_library(self):
        """
        Locates and loads the compiled C shared library from the ../bin directory.
        Handles OS-specific extentions (.dll vs .so).
        """
        # 1. Determine the file extention based on the OS
        system_name = platform.system()
        if system_name == "Windows":
            lib_name = "libaudio_engine.dll"
            self._log("DEBUG: HEY WINDOWS")
        elif system_name == "Darwin": # macOS
            lib_name = "libaudio_engine.dylib"
            self._log("DEBUG: HEY MACOS")
        else: # Linux
            lib_name = "libaudio_engine.so"
            self._log("DEBUG: HEY LINUX")

        # 2. Construct absolute path to the bin folder
        # Get the folder containing this script (src_py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level, then down into bin
        lib_path = os.path.join(current_dir, "..", "bin", lib_name)

        # #. Load the library
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find the compiled C library at: {lib_path}\nDid you run 'make'?")

        try:
            self._lib = ctypes.CDLL(lib_path)
            self._log(f"Successfully loaded audio engine: {lib_name}")
        except OSError as e:
            self._log(f"Error loading C library: {e}")
            sys.exit(1)

    def _configure_functions(self):
        """
        Defines the argument types (argtypes) and return types (restype)
        for the C functions. This prevents segmentation faults.
        """
        # --- Logging Function ---
        try:
            # --- void set_log_callback(logCallback callback) ---
            self._lib.set_log_callback.argtypes = [LOG_CALLBACK_TYPE]
            self._lib.set_log_callback.restype = None
        except AttributeError:
            self._log("Warning: set_log_callback not found in C library.")

        # --- Audio Control Functions ---
        try:
            # --- void init_audio() ---
            self._lib.init_audio.argtypes = []
            self._lib.init_audio.restype = None
            
            # --- void start_stream() ---
            self._lib.start_stream.argtypes = []
            self._lib.start_stream.restype = None
            
            # --- void stop_stream() ---
            self._lib.stop_stream.argtypes = []
            self._lib.stop_stream.restype = None
            
            # --- void load_impulse_response(const char* filepath) ---
            self._lib.load_impulse_response.argtypes = [ctypes.c_char_p]
            self._lib.load_impulse_response.restype = None
            
            # --- void set_dry_wet(float mix) ---
            self._lib.set_dry_wet.argtypes = [ctypes.c_float]
            self._lib.set_dry_wet.restype = None

            # --- void cleanup() ---
            self._lib.cleanup.argtypes = []
            self._lib.cleanup.restype = None

            # --- float get_cpu_load() ---
            self._lib.get_cpu_load.argtypes = []
            self._lib.get_cpu_load.restype = ctypes.c_float

        except AttributeError as e:
            self._log(f"Warning: Function missing in C library: {e}")
        
    # ===========================================================
    # Public API (The API called by GUI)
    # ===========================================================
    
    def register_logger(self, python_function):
        """
        Takes a Python function (which accepts a string) and passes it to C.
        """
        if self._lib:
            # 1. Wrap the Python function in the C-type definition
            # We assume python_function takes a byte string, we decode it inside the wrapper usually,
            # but ctypes passes bytes. making a lambda to decode it.

            def callback_handler(msg_bytes):
                msg = msg_bytes.decode('utf-8')
                python_function(msg)

            # create the C-callable function pointer
            c_callback = LOG_CALLBACK_TYPE(callback_handler)

            # 2. Store a reference so Python doesn't garbage collect it
            self._log_bc_ref = c_callback

            # 3. Pass to C
            self._lib.set_log_callback(c_callback)

    def initialize(self):
        """Initialize Miniaudio and FFTW resouces."""
        if self._lib:
            self._lib.init_audio()

    def start(self):
        """Starts the audio stream."""
        if self._lib:
            self._lib.start_stream()

    def stop(self):
        """Stops the audio stream."""
        if self._lib:
            self._lib.stop_stream()

    def load_ir_file(self, filepath):
        """
        Loads a wav file for convolution.
        Handles the string-to-bytes conversion required by C.
        """
        if self._lib and filepath:
            # C expects bytes, not a python string
            # .encode('utf-8') converts the string to bytes
            byte_path = filepath.encode('utf-8')
            self._lib.load_impulse_response(byte_path)

    def set_mix(self, value):
        """
        Value should be a float between 0.0 and 1.0
        """
        if self._lib:
            # ctypes.c_float ensures we pass a C-compatible float
            self._lib.set_dry_wet(ctypes.c_float(value))

    def close(self):
        """Frees memory and shuts down C resources."""
        if self._lib:
            self._lib.cleanup()

    def get_performance(self):
        """Returns CPU load as a percentage float."""
        if self._lib:
            return self._lib.get_cpu_load()
        return 0.0
## Build Instructions

### Prerequisites
This projects uses a C backend that must be compiled. You will need:
- **GCC Compiler**
- **Make**
- **Python 3.x**
- **FFTW3 Library**

### Linux
1. **Install Dependencies**:
You need `Build-essential` for GCC/Make and the development headers for FFTW3.
```bash
sudo apt-get update
sudo apt-get install build-essential libfftw3-dev
```

2. **Compile**:
```bash
make
```
This creates `bin/libaudio_engine.so`

### MacOS
1. **Install Homebrew** (if not already installed).
2. **Install FFTW**:
```bash
brew install fftw
```
Note: The Makefile is configured to look for headers in `/opt/homebrew/include` (Apple Silicon standard)

3. **Compile**:
```bash
make
```
This creates `bin/libaudio_engine.dylib`

### Windows
1. **Install a GCC Environment**: It is recommended to use **MSYS2** or **MinGW-w64**.
    - if using MSYS2, install dependencies via pacman:
    ```bash
    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make mingw-w64-x86_64-fftw
    ```
2. **Manual FFTW3 Setup (Alternative)**: If you are not using a package manager, you must download the precompiled FFTW DLLs:
    - Place `libfftw3-3.dll` in the `bin/` folder (so the app can find it at runtime).
    - Ensure the compiler can find the `.h` and `.a` (lib) files during the build (you may need to adjust the `Makefile` CFLAGS/LDFLAGS if they are not in your global path).
3. **Compile**: Open your terminal (CMD or PowerShell) in the project root and run:
```bash
make
```
This creates `bin/libaudio_engine.dll`

### Running the Application
Once compiled, set up the Python environment and run the wrapper:
```bash
pip install -r requirements.txt
python src_py/main.py
```

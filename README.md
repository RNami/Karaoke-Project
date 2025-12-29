## Handle External Dependencies (FFTW3)
- **Windows**: Linking against a local copy (`-I./include` and `bin/libfftw3-3.dll`)
- **Linux**: Command: `sudo apt-get install libfftw3-dev`
- **MacOS**: Command: `brew install fftw`

## How to compile:
1. Clone the repository
2. Install FFTW3 (using `apt` on Linux or `brew` on MacOS).
3. Run `make` in the terminal. This will generate the correct binary file inside the `bin/` folder for your specific OS.
4. Run `src_py/main.py`. The python wrapper will detect the OS and load the just build binary.

## How to run:
```bash
pip install -r requirements.txt
python ./src_py/main.py
```
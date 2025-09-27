"""
main.py
Entry point for the project.
"""
from realtime_conv import run_realtime_convolution


# EDIT: point to your IR
IR_PATH = "air_binaural_aula_carolina_0_1_3_180_3" \
".mat"

def main():
    # 1024-sample blocks; mic mono in; output channels = IR channels
    run_realtime_convolution(
        ir_path=IR_PATH,
        block=256,
        in_device_index=None,   # set an index to force; None = you'll be prompted
        out_device_index=None,  # set an index to force; None = you'll be prompted
        sample_rate=None,       # None = use device default; otherwise force (e.g., 44100)
        wet=1.0,
        dry=0.0
    )

if __name__ == "__main__":
    main()
 
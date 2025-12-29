/*
 * engine.c
 * The Core Audio Engine
 * * Responsibilities
 * 1. Initialize Audio Device (Miniaudio)
 * 2. Handle the real-time audio callback
 * 3. Expose control functions to Python
 * Features:
 * 1. Low-latency convolution for long IRs (partitioned)
 * 2. Automatic Resampling on load
 * 3. Stereo-to-mono downmix
 * 4. Robust ring buffering
*/

#define MINIAUDIO_IMPLEMENTATION
#include "../include/miniaudio.h"

// As I'm using miniaudio's decoding, raw dr_wav is not needed anymore.
// #define DR_WAV_IMPLEMENTATION
// #include "../include/dr_wav.h"

#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <time.h>
#endif

// --- DSP CONSTANTS ---
// LATENCY = BLOCK_SIZE / SAMPLE_RATE
// 1024 / 44100 = ~23 ms latency
#define BLOCK_SIZE 4096
#define FFT_SIZE (BLOCK_SIZE * 2) // To maximize FFTW efficiency, FFT_SIZE should be a power of 2
#define SAMPLE_RATE 44100
#define CHANNEL_COUNT 2
#define LOG_TAG "[C Engine]"
#define RING_BUFFER_SIZE (BLOCK_SIZE * 8) // size 4x block size for safety margin for thread decoupling

// Max number of partitions to prevent memory explosion
// 1024 samples * 500 partitions = ~500,000 samples = ~11 seconds max IR length
#define MAX_PARTITIONS 500

// --- Constants ---
#define SAMPLE_FORMAT ma_format_f32

// --- Logging Setup ---
typedef void (*LogCallback)(const char* message); // 1. Define the function signature (must match python's expectation)
static LogCallback logger = NULL; // 2. Global variable to store the python function

void log_msg(const char* format, ...) { // 3. Helper function to format strings and call Python
    if (logger == NULL) return;
    char message_buffer[1024];
    char final_buffer[1200];
    va_list args;
    va_start(args, format);
    vsnprintf(message_buffer, sizeof(message_buffer), format, args); // format the string safely (avoids buffer overflow)
    va_end(args);
    snprintf(final_buffer, sizeof(final_buffer), "%s %s", LOG_TAG, message_buffer); // prepend the tag automatically
    logger(final_buffer); // call python here
}

void set_log_callback(LogCallback callback) { // handshake, // 4. Exported function: python calls this to "register" itself
    logger = callback;
    log_msg("Logger registered successfully.\n");
}

double get_time_seconds() {
    #ifdef _WIN32
        LARGE_INTEGER freq, count;
        QueryPerformanceFrequency(&freq); // 1. How fast does the clock tick
        QueryPerformanceCounter(&count); // 2. What tick are we on now
        return (double)count.QuadPart / freq.QuadPart;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts); // time since the system booted
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    #endif
}

// --- Audio Engine State ---
// This holds everything out audio engine needs to remember between callbacks
typedef struct {
    ma_device device; // The handle to the physical audio hardware
    ma_device_config deviceConfig; // Settings used to set up the device (rate, channels, etc.)

    // Control Parameters
    volatile float dry_wet_mix; // 0.0 = Dry, 1.0 Wet
    volatile int is_filter_loaded; // Flag to check if an audio effect is currently active
    volatile double current_cpu_load; // Performance metric (0.0 to 1.0+)

    // --- FFTW Resources ---
    fftw_plan fft_plan;
    fftw_plan ifft_plan;

    double* fft_in_buffer; // Real input [FFT_SIZE]
    fftw_complex* fft_out_buffer; // complex spectrum [FFT_SIZE/2 + 1]
    double* ifft_result_buffer; // Real output [FFT_SIZE]

    // --- Partitioned Convolution State ---
    int num_partitions;
    // Array of pointers to spectra. Partitions[0] is the first chunk of the IR, etc.
    fftw_complex* ir_partitions[MAX_PARTITIONS];

    // Frequency Delay Line (FDL)
    // A ring buffer of input spectra. fdl[0] is current input spectrum, fdl[1] is previous, etc.
    fftw_complex* fdl[MAX_PARTITIONS];
    int fdl_index; // Current write head for the FDL

    // --- Audio Buffering (Ring Buffer) ---
    // collecting samples until we have BLOCK_SIZE
    float* input_ring_buffer;
    float* output_ring_buffer;
    int ring_write_pos;
    int ring_read_pos;
    int samples_in_buffer;

    // --- Overlap_Add State ---
    double* overlap_buffer; // tail of the previous block

    // TODO: add FFT buffers here later
    // ...
} AudioEngine;

static AudioEngine engine = {0}; // global instance
// TODO: This is just a simple use case, might change to pass handles later

// ===========================================
// --- INTERNAL DSP FUNCTIONS ---
// ===========================================

void init_dsp_buffers() {
    // 1. FFTW Allocations
    // Allocate FFTW arrays (must be aligned -> fftw_malloc)
    engine.fft_in_buffer = fftw_alloc_real(FFT_SIZE);
    engine.ifft_result_buffer = fftw_alloc_real(FFT_SIZE);
    engine.fft_out_buffer = fftw_alloc_complex(FFT_SIZE / 2 + 1);

    // 2. Partition Allocations
    for(int i=0; i<MAX_PARTITIONS; i++) {
        engine.ir_partitions[i] = fftw_alloc_complex(FFT_SIZE / 2 + 1);
        engine.fdl[i] = fftw_alloc_complex(FFT_SIZE / 2 + 1);
        
        // Zero out to prevent garbage noise
        memset(engine.ir_partitions[i], 0, sizeof(fftw_complex) * (FFT_SIZE/2 + 1));
        memset(engine.fdl[i], 0, sizeof(fftw_complex) * (FFT_SIZE/2 + 1));
    }

    // 3. Plans
    // create plans (heavy calculation, do once)
    // FFT_ESTIMATE is faster to initialize, FFT_MEASURE is faster to run but slower to init
    engine.fft_plan = fftw_plan_dft_r2c_1d(FFT_SIZE, engine.fft_in_buffer, engine.fft_out_buffer, FFTW_ESTIMATE); // TODO: Maybe change to FFTW_ESTIMATE depending on the result in testing
    engine.ifft_plan = fftw_plan_dft_c2r_1d(FFT_SIZE, engine.fft_out_buffer, engine.ifft_result_buffer, FFTW_ESTIMATE);

    // 4. Time Domain Buffers
    // Buffers
    engine.overlap_buffer = (double*)calloc(FFT_SIZE, sizeof(double));

    // simple ring buffer (size 4x block size for safety)
    int ring_size = RING_BUFFER_SIZE;
    engine.input_ring_buffer = (float*)calloc(ring_size, sizeof(float));
    engine.output_ring_buffer = (float*)calloc(ring_size, sizeof(float));

    engine.ring_write_pos = 0;
    engine.ring_read_pos = 0;
    engine.samples_in_buffer = 0;
    engine.fdl_index = 0;
    engine.num_partitions = 0;

    log_msg("DSP Initialized. Block: %d, FFT: %d, Max Partitions: %d\n", BLOCK_SIZE, FFT_SIZE, MAX_PARTITIONS);
}

// COMPLEX MULTIPLY ACCUMULATE
// acc += a * b
void complex_mac(fftw_complex acc, fftw_complex a, fftw_complex b) {
    // (ar + j*ai) * (br + j*bi) = (ar*br - ai*bi) + j*(ar*bi + ai*br)
    double real = (a[0] * b[0]) - (a[1] * b[1]);
    double imag = (a[0] * b[1]) + (a[1] * b[0]);
    acc[0] += real;
    acc[1] += imag;
}

void process_block(float* input_block, float* output_block) {
    // 1. Prepare FFT input: copy BLOCK_SIZE samples, pad rest with zeros
    for (int i = 0; i < BLOCK_SIZE; i++)
        engine.fft_in_buffer[i] = (double)input_block[i];
    for (int i = BLOCK_SIZE; i < FFT_SIZE; i++)
        engine.fft_in_buffer[i] = 0.0;

    // 2. FFT current block
    fftw_execute(engine.fft_plan);

    // 3. Save to Frequency Delay Line (FDL)
    // then we overwrite the oldest block in the circular buffer
    memcpy(engine.fdl[engine.fdl_index], engine.fft_out_buffer, sizeof(fftw_complex) * (FFT_SIZE/2 + 1));

    // 4. Partitioned Convolution (the "Sum" loop)
    // Clear accumulator (reusing fft_out_buffer as accumulator)
    memset(engine.fft_out_buffer, 0, sizeof(fftw_complex) * (FFT_SIZE/2 + 1));

    // Math: Output += FDL[past_index] * Filter_Partition[k]
    for (int k = 0; k < engine.num_partitions; k++) {
        // Find the past input block corresponding to this IR partition
        int fdl_pos = (engine.fdl_index - k + MAX_PARTITIONS) % MAX_PARTITIONS;

        // Accumulate: Output += FDL[past] * IR[k]
        for (int bin = 0; bin < FFT_SIZE / 2 + 1; bin++) {
            complex_mac(engine.fft_out_buffer[bin], engine.fdl[fdl_pos][bin], engine.ir_partitions[k][bin]);
        }
    }

    // Move FDL head
    engine.fdl_index = (engine.fdl_index + 1) % MAX_PARTITIONS;

    // 5. IFFT
    fftw_execute(engine.ifft_plan);

    // 6. Overlap-Add & Output
    double scale = 1.0 / (double)FFT_SIZE; // for normalization

    float wet_gain = engine.dry_wet_mix;
    float dry_gain = 1.0f - wet_gain;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        double raw_val = engine.ifft_result_buffer[i] * scale;

        // Add overlap
        raw_val += engine.overlap_buffer[i];

        // Output mix
        output_block[i] = (input_block[i] * dry_gain) + ((float)raw_val * wet_gain);

        // Clear used overlap
        engine.overlap_buffer[i] = 0.0f;
    }
    // Handle tail (Save for next block)
    for (int i = BLOCK_SIZE; i < FFT_SIZE; i++) {
        double raw_val = engine.ifft_result_buffer[i] * scale;

        // Add to existing overlap (Standard Overlap-Add logic)
        engine.overlap_buffer[i - BLOCK_SIZE] += raw_val;
    }
}

// ===========================================
//  AUDIO CALLBACK
// * runs on a high-priority background thread. No printf calls here ideally!
    //* CRITICAL: NEVER call log_msg() here!
    //* It is too slow for real-time audio and will cause crackling/glitches
// ===========================================
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    double start_time = get_time_seconds();
    AudioEngine* eng = (AudioEngine*)pDevice->pUserData;

    // Cast void* to float* for arithmetic
    float* in = (float*)pInput;
    float* out = (float*)pOutput;
    // framecount is the number of "sample pairs" (L+R)
    // Total float count = frameCount * CHANNEL_COUNT
    int total_samples = frameCount * CHANNEL_COUNT;

    // --- PROCESSING LOGIC ---
    if (eng->is_filter_loaded) {
        // COMPLEX PATH: Convolution (To be implemented)
        // For now, silence or fallback
        memset(out, 0, total_samples * sizeof(float));
    } else {
        // SIMPLE PATH: Passthrough (Copy Input to Output)
        // This confirms the audio pipeline works
        memcpy(out, in, total_samples * sizeof(float));
    }

    double end_time = get_time_seconds();
    double process_time = end_time - start_time;
    double max_time_available = (double)frameCount / (double)SAMPLE_RATE;
    // Load = time taken / time available, e.g., if processing takes 2ms and we have 10ms buffer, load is 0.2 (20%)
    eng->current_cpu_load = process_time / max_time_available;
}

// ===========================================
// --- EXPORTED FUNCTIONS (Called from Python) ---
// ===========================================

// 1. Initialize the Audio System
void init_audio() {
    log_msg("Initializing Miniaudio...\n");

    engine.deviceConfig = ma_device_config_init(ma_device_type_duplex);
    engine.deviceConfig.capture.pDeviceID = NULL; // Default mic
    engine.deviceConfig.capture.format = SAMPLE_FORMAT;
    engine.deviceConfig.capture.channels = CHANNEL_COUNT;
    engine.deviceConfig.playback.pDeviceID = NULL; // Default Speakers
    engine.deviceConfig.playback.format = SAMPLE_FORMAT;
    engine.deviceConfig.playback.channels = CHANNEL_COUNT;
    engine.deviceConfig.sampleRate = SAMPLE_RATE;
    engine.deviceConfig.dataCallback = data_callback;
    engine.deviceConfig.pUserData = &engine;

    if (ma_device_init(NULL, &engine.deviceConfig, &engine.device) != MA_SUCCESS) {
        // fprintf(stderr, "[C Engine] Failed to initialize device.\n");
        log_msg("Failed to initialize device.\n");
        return;
    }

    // Set defaults
    engine.dry_wet_mix = 0.0f;
    engine.is_filter_loaded = 0;

    log_msg("Audio initialized successfully. Rate: %d\n", engine.deviceConfig.sampleRate);
}

// 2. Start the Stream
void start_stream() {
    if (ma_device_start(&engine.device) != MA_SUCCESS) {
        // fprintf(stderr, "[C Engine] Failed to start device.\n");
        log_msg("Failed to start device.\n");
    } else {
        log_msg("Stream Started.\n");
    }
}

// 3. Stop the Stream
void stop_stream() {
    if (ma_device_stop(&engine.device) != MA_SUCCESS) {
        // fprintf(stderr, "[C Engine] Failed to stop device.\n");
        log_msg("Failed to stop device.\n");
    }
}

void load_impulse_response(const char* filepath) {
    log_msg("Loading IR file: %s\n", filepath);

    // TODO:
    // 1. load WAV file using dr_wav
    // 2. Pad to FFT size
    // 3. Compute FFT of the IR
    // 4. Update engine.is_filter_loaded = 1

    // For now, we just acknowledge receipt for debug
    engine.is_filter_loaded = 0; // Keep 0 so we don't trigger empty processing logic
}

void set_dry_wet(float mix) {
    if (mix < 0.0f) mix = 0.0f;
    if (mix > 1.0f) mix = 1.0f;
    engine.dry_wet_mix = mix;
    // log_msg("Mix set to %.2f\n", mix); // Commented out for now to avoid spamming stdout
}

// 6. Cleanup
void cleanup() {
    ma_device_uninit(&engine.device);
    log_msg("Resources freed.\n");
}

// getter function for performance stats
float get_cpu_load() {
    return (float)(engine.current_cpu_load * 100.0); // Convert to percentage
}

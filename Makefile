CC = gcc
TARGET_NAME_RAW = libaudio_engine

ifeq ($(OS),Windows_NT)
    # --- WINDOWS SETTINGS ---
    TARGET_NAME = $(TARGET_NAME_RAW).dll
    TARGET = bin/$(TARGET_NAME)
    
    CC = gcc
    CFLAGS = -Wall -O3 -fPIC -I./include
    # Linking for Windows
    LDFLAGS = -shared -lfftw3 -lm
    
    MKDIR_CMD = if not exist bin mkdir bin
    RM_CMD = del /Q /F
    CLEAN_TARGETS = bin\$(TARGET_NAME_RAW).so bin\$(TARGET_NAME_RAW).dll bin\$(TARGET_NAME_RAW).dylib

else
    # --- LINUX / MACOS SETTINGS ---
    UNAME_S := $(shell uname -s)
    CFLAGS = -Wall -O3 -fPIC
    MKDIR_CMD = mkdir -p bin
    RM_CMD = rm -f
    CLEAN_TARGETS = bin/$(TARGET_NAME_RAW).so bin/$(TARGET_NAME_RAW).dll bin/$(TARGET_NAME_RAW).dylib

    ifeq ($(UNAME_S),Darwin)
        # macOS
        TARGET_NAME = $(TARGET_NAME_RAW).dylib
        TARGET = bin/$(TARGET_NAME)
        CFLAGS += -I/opt/homebrew/include
        LDFLAGS = -shared -L/opt/homebrew/lib -lfftw3 -lm -lpthread
    else
        # Linux
        TARGET_NAME = $(TARGET_NAME_RAW).so
        TARGET = bin/$(TARGET_NAME)
        LDFLAGS = -shared -lfftw3 -lm -lpthread
    endif
endif

SOURCES = src_c/engine.c

all: directories $(TARGET)

directories:
	$(MKDIR_CMD)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(RM_CMD) $(CLEAN_TARGETS)
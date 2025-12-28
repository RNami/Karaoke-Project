# CC = gcc

ifeq ($(OS),Windows_NT)
    # --- WINDOWS SETTINGS ---
    TARGET_NAME = libaudio_engine.dll
    TARGET = bin/$(TARGET_NAME)
    
    CC = gcc
    CFLAGS = -Wall -O3 -fPIC -I./include
    # Linking for Windows
    LDFLAGS = -shared -lfftw3 -lm
    
    MKDIR_CMD = if not exist bin mkdir bin
    RM_CMD = del /Q /F
    CLEAN_TARGETS = bin\*.dll bin\*.so bin\*.dylib

else
    # --- LINUX / MACOS SETTINGS ---
    UNAME_S := $(shell uname -s)
    CFLAGS = -Wall -O3 -fPIC
    MKDIR_CMD = mkdir -p bin
    RM_CMD = rm -f
    CLEAN_TARGETS = bin/*.so bin/*.dll bin/*.dylib

    ifeq ($(UNAME_S),Darwin)
        # macOS
        TARGET_NAME = libaudio_engine.dylib
        TARGET = bin/$(TARGET_NAME)
        CFLAGS += -I/opt/homebrew/include
        LDFLAGS = -shared -L/opt/homebrew/lib -lfftw3 -lm -lpthread
    else
        # Linux
        TARGET_NAME = libaudio_engine.so
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
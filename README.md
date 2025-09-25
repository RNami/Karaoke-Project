# Block A: RT Playback
## Member:
- Bhavya
- Philip
- Samu
## Step 0: Real Time Playback

- [ ] Research for realtime playback options in Python
- [ ] Find a solution for select the audio input/output
- [ ] provide an array to process the input audio in Python
- [ ] playback the data again on the output device
- [ ] test it with the audio interface


# Block B: Room Simulation Processing

## Member:
- Khashayar
- Joengjoon
## Step 1: Import data

- [ ] import any  audio test data (only for testing the script/processing)
- [ ] Import the Room impulse response from the data set
	- [ ] use scipy.io.loadmat()
- [ ] bring the data into a desired format for later processing (np.array?)

## Step 2: Convolution Implementation

- [ ] research for liabriries to filter input test audio file with the IR
- [ ] do we have to filter it first?
- [ ] implement a convolution with the RIR 
	- [ ] different implemetation to compare, w.r.t.
		- [ ] Runtime
		- [ ] RT feasibility?
		- [ ] memory?

# Block C: Combining
## Step 3: Real Time Convolution
- [ ] Combine both blocks, s.t. a realtime processing is possible
## Step 4: Sound Quality Check

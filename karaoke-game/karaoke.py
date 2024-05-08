import pyaudio
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy import signal

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
p = pyaudio.PyAudio()

kernel_size = 20
kernel_sigma = 1
sampling_rate = 44100 #https://de.wikipedia.org/wiki/Abtastrate

# print info about audio devices
# let user select audio device
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

print('select audio device:')
input_device = int(input())

# open audio input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=input_device)

# set up interactive plot
fig = plt.figure()
ax = plt.gca()
line, = ax.plot(np.zeros(CHUNK_SIZE))
line2, = ax.plot(np.zeros(CHUNK_SIZE))
ax.set_ylim(-30000, 30000)

plt.ion()
plt.show()

def apply_kernel(data):
    kernel = signal.windows.gaussian(kernel_size, kernel_sigma) # create a kernel
    kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
    return np.convolve(data, kernel, 'same') # apply the kernel to the signal

def apply_hamming_window(data):
    hamming_window = np.hamming(CHUNK_SIZE)
    return data * hamming_window

def get_max_frequency(data) -> float:
    # calculate spectrum using a fast fourier transform
    spectrum = np.abs(np.fft.fft(data))
    
    # resample x axis of spectrum to match frequency even if sample_length != 1
    frequencies = np.fft.fftfreq(CHUNK_SIZE, 1/sampling_rate)
    
    # get rid of negative half
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    spectrum = spectrum[mask]

    #get max frequency
    max_frequency = positive_frequencies[np.argmax(spectrum)]
    return max_frequency

# this method was generated with GPT
def get_note(max_frequency) -> str:
    print(max_frequency)
    if max_frequency:
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        A4_frequency = 440 #https://mixbutton.com/mixing-articles/music-note-to-frequency-chart/
        # calculate half tones from A4 to max_frequenca
        half_steps = int(round(12 * np.log2(max_frequency / A4_frequency)))
        # find the note
        note_index = (half_steps + 9) % 12
        note = notes[note_index]
        return note
    return ""

# continuously capture and plot audio signal
while True:
    # Read audio data from stream
    #data = stream.read(CHUNK_SIZE)
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)
    #print(data)

    data = apply_kernel(data)

    data = apply_hamming_window(data)

    max_frequency = get_max_frequency(data)
    note = get_note(max_frequency)
    print(note)


    line.set_ydata(data)

    # Redraw plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    # plt.plot(frequencies[mask], spectrum[mask])
    # plt.legend()
    # plt.show()

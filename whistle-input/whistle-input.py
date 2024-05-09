import pyaudio
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy import signal
import pyglet
from pyglet import shapes, clock


# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
p = pyaudio.PyAudio()

A4_FREQUENCY = 440 #https://mixbutton.com/mixing-articles/music-note-to-frequency-chart/

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 650
RECT_HEIGHT = 50
RECT_WIDTH = 400
NOTE_WIDTH_MULTIPLIER = 100
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)


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

def get_current_frequency() -> int:
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    data = apply_kernel(data)
    data = apply_hamming_window(data)
    max_frequency = get_max_frequency(data)
    return max_frequency

class Rectangle:

    def __init__(self, y_position):
        self.rect = shapes.Rectangle(x=WINDOW_WIDTH//2 - RECT_WIDTH//2, y=y_position, width=RECT_WIDTH, height=RECT_HEIGHT, color=(0, 0, 255))
        self.selected = False

    def select(self):
        self.selected = True
        self.rect.color = (72, 209, 204)

    def deselect(self):
        self.selected = False
        self.rect.color = (0, 0, 255)

    def draw(self):
        self.rect.draw()

rectangles = []

for i in range(0, 9):
    rect = Rectangle(i * 75)
    rectangles.append(rect)

rectangles[4].select()

def rect_up():
    for index, rect in enumerate(rectangles):
        if(rect.selected):
            if(index + 1 < len(rectangles)):
                rect.deselect()
                rectangles[index + 1].select()
                return

def rect_down():
    for index, rect in enumerate(rectangles):
        if(rect.selected):
            if(index - 1 >= 0):
                rect.deselect()
                rectangles[index - 1].select()
                return


counter_up = 0
counter_down = 0

old_fr = -1

#these variables prevent from detecting the same sequence twice
up_captured = False 
down_captured = False

# continuously capture and plot audio signal
def whistle_input():
    global old_fr, up_captured, down_captured, counter_up, counter_down
    fr = get_current_frequency()
    if(old_fr < fr and not up_captured):
        down_captured = False
        counter_up += 1
        counter_down = 0
        if(counter_up > 7):
            print("UP INPUT")
            rect_up()
            counter_up = 0
            up_captured = True
    elif(old_fr > fr and not down_captured):
        up_captured = False
        counter_down += 1
        counter_up = 0
        if(counter_down > 7):
            print("DOWN INPUT")
            rect_down()
            counter_down = 0
            down_captured = True
    old_fr = fr

@window.event
def on_draw():
    whistle_input()
    window.clear()
    for rect in rectangles:
        rect.draw()   


pyglet.app.run()
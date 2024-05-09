import pyaudio
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy import signal
import pyglet
from pyglet import shapes, clock
import sys
sys.path.append('../read_midi')
from read_midi import get_midi_data, get_note_from_midi
from os.path import dirname, join

#USES CODE FROM audio-sample.py AND dsp-solution.ipynb

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
p = pyaudio.PyAudio()

A4_FREQUENCY = 440 #https://mixbutton.com/mixing-articles/music-note-to-frequency-chart/
OCTAVE_SIZE = 12
MIN_LOUDNESS = 750

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 635 # = max midi notes * 5
RECT_HEIGHT = 5
NOTE_WIDTH_MULTIPLIER = 100
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

KERNEL_SIZE = 20
KERNEL_SIGMA = 1

current_directory = dirname(__file__)
midi_file_path = join(current_directory, '../read_midi/freude.mid')
#midi_file_path = join(current_directory, '../read_midi/berge.mid')
midi_data = get_midi_data(midi_file_path)

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
    kernel = signal.windows.gaussian(KERNEL_SIZE, KERNEL_SIGMA) # create a kernel
    kernel /= np.sum(kernel) # normalize the kernel so it does not affect the signal's amplitude
    return np.convolve(data, kernel, 'same') # apply the kernel to the signal

def apply_hamming_window(data):
    hamming_window = np.hamming(CHUNK_SIZE)
    return data * hamming_window

def get_max_frequency(data) -> float:
    # calculate spectrum using a fast fourier transform
    spectrum = np.abs(np.fft.fft(data))
    
    # resample x axis of spectrum to match frequency even if sample_length != 1
    frequencies = np.fft.fftfreq(CHUNK_SIZE, 1/RATE)
    
    # get rid of negative half
    mask = frequencies >= 0
    positive_frequencies = frequencies[mask]
    spectrum = spectrum[mask]

    #get max frequency
    max_frequency = positive_frequencies[np.argmax(spectrum)]
    return max_frequency

# this method was generated with GPT
def get_note(max_frequency) -> str:
    if max_frequency:
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # calculate half tones from A4 to max_frequenca
        half_steps = int(round(OCTAVE_SIZE * np.log2(max_frequency / A4_FREQUENCY)))
        # find the note
        note_index = (half_steps + 9) % OCTAVE_SIZE
        note = notes[note_index]
        return note
    return ""

def get_current_note() -> str:
    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    if(data[np.argmax(data)] < MIN_LOUDNESS):
        return "" #voice is not loud enough, avoid background sounds

    data = apply_kernel(data)
    data = apply_hamming_window(data)
    max_frequency = get_max_frequency(data)
    note = get_note(max_frequency)
    return note

#class for one note block in the display
class NoteBlock:

    def __init__(self, midi_note, duration, x_position):
        self.rect = shapes.Rectangle(x=WINDOW_WIDTH + x_position, y=midi_note * RECT_HEIGHT, width=duration * NOTE_WIDTH_MULTIPLIER, height=RECT_HEIGHT, color=(255, 0, 0))
        self.note = get_note_from_midi(midi_note)
        self.hit_note = False

    def move_left(self):
        self.rect.x = self.rect.x - 1

    def change_to_green(self):
        self.hit_note = True
        self.rect.color = (0, 255, 0)

    def draw(self):
        self.rect.draw()

    def check_note(self):
        if(not self.hit_note):
            if(self.rect.x < WINDOW_WIDTH // 3 < self.rect.x + self.rect.width): #note block crosses line
                if(get_current_note() == self.note):
                    self.change_to_green()

def calculate_duration(midi_data, index):
    length = 0
    for i in range(index + 1, len(midi_data)):
        length += midi_data[i].time
        if(midi_data[i].type == "note_off"):
            if(midi_data[i].note == midi_data[index].note):
                return length#round(length, 3)
    return length

blocks = []

def print_block(index, data, midi_data, x_position):
    if(data.type == "note_on"):
        block = NoteBlock(data.note, calculate_duration(midi_data, index), x_position)
        blocks.append(block)

def move_blocks(dt):
    for block in blocks:
        block.move_left()

line_x = WINDOW_WIDTH // 3
line = shapes.Line(line_x, 0, line_x, WINDOW_HEIGHT, color=(255, 255, 255))

@window.event
def on_draw():
    window.clear()
    line.draw()
    for block in blocks:
        block.check_note()
        block.draw()   

@window.event       
def on_activate():
    clock.tick(5)
    actual_time = 0
    for index, data in enumerate(midi_data):
        actual_time += data.time
        x_position = actual_time * NOTE_WIDTH_MULTIPLIER
        print_block(index, data, midi_data, x_position)

clock.schedule_interval(move_blocks, 1/NOTE_WIDTH_MULTIPLIER) #https://pyglet.readthedocs.io/en/latest/modules/clock.html

pyglet.app.run()
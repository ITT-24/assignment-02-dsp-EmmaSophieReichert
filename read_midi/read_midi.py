import mido
from mido import MidiFile

# for msg in MidiFile('freude.mid').play():
#     print(msg)
#     #print(msg.note)
#     #print(msg.type)

def get_midi_data(midi_file_path) -> list:
    midi_data = []
    for msg in MidiFile(midi_file_path ).play():
        midi_data.append(msg)
    return midi_data

def get_note_from_midi(midi_note) -> str:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_index = midi_note % 12
    return notes[note_index]


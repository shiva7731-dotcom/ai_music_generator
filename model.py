import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream

def get_training_notes(midi_folder="midi_songs"):
    notes = []
    
    # Get the absolute path of the directory where model.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(base_dir, midi_folder)
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created '{midi_folder}' directory at: {target_folder}")
        print("Please place your MIDI files there and run again.")
        return notes

    print(f"Scanning folder: {target_folder}...")
    for file in os.listdir(target_folder):
        if file.lower().endswith((".mid", ".midi")):
            file_path = os.path.join(target_folder, file)
            print(f"Parsing: {file}...")
            try:
                midi = converter.parse(file_path)
                parts = instrument.partitionByInstrument(midi)
                notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            except Exception as e:
                print(f"Error parsing {file}: {e}")
                
    return notes

def prepare_sequences(notes, seq_length=100):
    pitch_names = sorted(set(item for item in notes))
    pitches_to_int = {note: number for number, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - seq_length, 1):
        sequence_in = notes[i:i + seq_length]
        sequence_out = notes[i + seq_length]
        network_input.append([pitches_to_int[char] for char in sequence_in])
        network_output.append(pitches_to_int[sequence_out])

    n_patterns = len(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, seq_length, 1))
    normalized_input = normalized_input / float(len(pitch_names))
    
    from tensorflow.keras.utils import to_categorical
    network_output = to_categorical(network_output)

    return normalized_input, network_output, pitch_names, pitches_to_int

def create_lstm_model(input_shape, n_vocab):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

    model = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_midi_from_model(model, network_input, pitch_names, output_file="output.mid"):
    int_to_pitch = {number: note for number, note in enumerate(pitch_names)}
    start_index = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start_index]
    
    prediction_output = []
    print("Generating new music sequence...")
    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitch_names))
        
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_pitch[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern, index)
        pattern = pattern[1:]

    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            chord_notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
            new_chord = chord.Chord(chord_notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"Successfully generated and saved MIDI file to '{output_file}'!")

if __name__ == "__main__":
    print("=" * 65)
    print("🎵 AI Music Generation Pipeline (Task 3: music21 & LSTM)")
    print("=" * 65)
    
    # Step 1: Get training notes from MIDI files
    notes = get_training_notes("midi_songs")
    
    if len(notes) > 0:
        print(f"Total notes found: {len(notes)}")
        
        # Step 2: Prepare sequences
        network_input, network_output, pitch_names, pitches_to_int = prepare_sequences(notes)
        
        # Step 3: Create model
        model = create_lstm_model(input_shape=(network_input.shape[1], network_input.shape[2]), n_vocab=len(pitch_names))
        
        # Step 4: Train model (uncomment fit when ready, e.g., 20 epochs for quick test)
        print("Training LSTM model...")
        model.fit(network_input, network_output, epochs=2, batch_size=64)
        
        # Step 5: Generate music
        generate_midi_from_model(model, network_input, pitch_names, output_file="generated_output.mid")
    else:
        print("\n⚠️ No MIDI files found in 'midi_songs/' folder!")
        print("Please add at least one .mid file into the 'midi_songs' directory and run again.")
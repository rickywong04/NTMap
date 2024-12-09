#!/usr/bin/env python3
import sys
import os
import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "models/saved_model.h5"
LABELS_PATH = "models/label_classes.npy"

SR = 44100
DURATION = 2.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

NOTE_NAME_MAP = {
    "E2": ("E2", "N/A"),
    "F2": ("F2", "E♯2"),
    "Fsharp2": ("F#2", "G♭2"),
    "G2": ("G2", "N/A"),
    "Gsharp2": ("G#2", "A♭2"),
    "A2": ("A2", "N/A"),
    "Asharp2": ("A#2", "B♭2"),
    "B2": ("B2", "C♭3"),
    "C3": ("C3", "B♯2"),
    "Csharp3": ("C#3", "D♭3"),
    "D3": ("D3", "N/A"),
    "Dsharp3": ("D#3", "E♭3"),
    "E3": ("E3", "N/A"),
    "F3": ("F3", "E♯3"),
    "Fsharp3": ("F#3", "G♭3"),
    "G3": ("G3", "N/A"),
    "Gsharp3": ("G#3", "A♭3"),
    "A3": ("A3", "N/A"),
    "Asharp3": ("A#3", "B♭3"),
    "B3": ("B3", "C♭4"),
    "C4": ("C4", "B♯3"),
    "Csharp4": ("C#4", "D♭4"),
    "D4": ("D4", "N/A"),
    "Dsharp4": ("D#4", "E♭4"),
    "E4": ("E4", "N/A"),
    "F4": ("F4", "E♯4"),
    "Fsharp4": ("F#4", "G♭4"),
    "G4": ("G4", "N/A"),
    "Gsharp4": ("G#4", "A♭4"),
    "A4": ("A4", "N/A"),
    "Asharp4": ("A#4", "B♭4"),
    "B4": ("B4", "C♭5"),
    "C5": ("C5", "B♯4"),
    "Csharp5": ("C#5", "D♭5"),
    "D5": ("D5", "N/A"),
    "Dsharp5": ("D#5", "E♭5"),
    "E5": ("E5", "N/A"),
    "F5": ("F5", "E♯5"),
    "Fsharp5": ("F#5", "G♭5"),
    "G5": ("G5", "N/A"),
    "Gsharp5": ("G#5", "A♭5"),
    "A5": ("A5", "N/A"),
    "Asharp5": ("A#5", "B♭5"),
    "B5": ("B5", "C♭6"),
    "C6": ("C6", "B♯5"),
    "Csharp6": ("C#6", "D♭6"),
    "D6": ("D6", "N/A"),
    "Dsharp6": ("D#6", "E♭6"),
    "E6": ("E6", "N/A")
}


NOTE_TO_TAB = {
    "E2": [(6, 0)],
    "F2": [(6, 1)],
    "Fsharp2": [(6, 2)],
    "G2": [(6, 3)],
    "Gsharp2": [(6, 4)],
    "A2": [(6, 5), (5, 0)],
    "Asharp2": [(6, 6), (5, 1)],
    "B2": [(6, 7), (5, 2)],
    "C3": [(6, 8), (5, 3)],
    "Csharp3": [(6, 9), (5, 4)],
    "D3": [(6, 10), (5, 5), (4, 0)],
    "Dsharp3": [(6, 11), (5, 6), (4, 1)],
    "E3": [(6, 12), (5, 7), (4, 2)],
    "F3": [(5, 8), (4, 3)],
    "Fsharp3": [(5, 9), (4, 4)],
    "G3": [(5, 10), (4, 5), (3, 0)],
    "Gsharp3": [(5, 11), (4, 6), (3, 1)],
    "A3": [(5, 12), (4, 7), (3, 2)],
    "Asharp3": [(5, 13), (4, 8), (3, 3)],
    "B3": [(5, 14), (4, 9), (3, 4)],
    "C4": [(5, 15), (4, 10), (3, 5), (2, 1)],
    "Csharp4": [(5, 16), (4, 11), (3, 6), (2, 2)],
    "D4": [(5, 17), (4, 12), (3, 7), (2, 3)],
    "Dsharp4": [(5, 18), (4, 13), (3, 8), (2, 4)],
    "E4": [(5, 19), (4, 14), (3, 9), (2, 5), (1, 0)],
    "F4": [(5, 20), (4, 15), (3, 10), (2, 6), (1, 1)],
    "Fsharp4": [(5, 21), (4, 16), (3, 11), (2, 7), (1, 2)],
    "G4": [(5, 22), (4, 17), (3, 12), (2, 8), (1, 3)],
    "Gsharp4": [(5, 23), (4, 18), (3, 13), (2, 9), (1, 4)],
    "A4": [(5, 24), (4, 19), (3, 14), (2, 10), (1, 5)],
    "Asharp4": [(4, 20), (3, 15), (2, 11), (1, 6)],
    "B4": [(4, 21), (3, 16), (2, 12), (1, 7)],
    "C5": [(4, 22), (3, 17), (2, 13), (1, 8)],
    "Csharp5": [(4, 23), (3, 18), (2, 14), (1, 9)],
    "D5": [(4, 24), (3, 19), (2, 15), (1, 10)],
    "Dsharp5": [(3, 20), (2, 16), (1, 11)],
    "E5": [(3, 21), (2, 17), (1, 12)],
    "F5": [(3, 22), (2, 18), (1, 13)],
    "Fsharp5": [(3, 23), (2, 19), (1, 14)],
    "G5": [(3, 24), (2, 20), (1, 15)],
    "Gsharp5": [(2, 21), (1, 16)],
    "A5": [(2, 22), (1, 17)],
    "Asharp5": [(2, 23), (1, 18)],
    "B5": [(2, 24), (1, 19)],
    "C6": [(1, 20)],
    "Csharp6": [(1, 21)],
    "D6": [(1, 22)],
    "Dsharp6": [(1, 23)],
    "E6": [(1, 24)]
}




def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True, duration=DURATION)
    # Ensure length
    required_length = int(SR*DURATION)
    if len(y) < required_length:
        y = np.pad(y, (0, required_length - len(y)), mode='constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, 
                                       hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db + 80.0) / 80.0
    S_norm = np.expand_dims(S_norm, axis=-1)  # (128, time_frames, 1)
    return S_norm

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_tabs.py <audio_file.wav>")
        sys.exit(1)
    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(LABELS_PATH, allow_pickle=True)

    X = extract_mel_spectrogram(audio_file)
    X = np.expand_dims(X, axis=0)  # (1, 128, time_frames, 1)

    predictions = model.predict(X)
    pred_class = classes[np.argmax(predictions)]

    if pred_class in NOTE_TO_TAB:
        # Get user-friendly names
        sharp_name, flat_name = NOTE_NAME_MAP.get(pred_class, (pred_class, "N/A"))

        mappings = NOTE_TO_TAB[pred_class]
        print(f"Predicted Note: {sharp_name} ({flat_name})")
        print("Possible Tabs:")
        for string, fret in mappings:
            print(f"  - String {string}, Fret {fret}")
    else:
        print(f"Predicted Note: {pred_class}")
        print("No tab mapping available for this note.")


if __name__ == "__main__":
    main()

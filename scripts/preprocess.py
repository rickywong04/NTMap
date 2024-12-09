#!/usr/bin/env python3
import os
import numpy as np
import librosa

RAW_DATA_DIR = "data/raw/Guitar Dataset"
OUTPUT_NPZ = "data/processed/processed_data.npz"

SR = 44100  # As stated, dataset is at 44.1 kHz
DURATION = 2.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True, duration=DURATION)
    # Ensure length
    required_length = int(SR*DURATION)
    if len(y) < required_length:
        y = np.pad(y, (0, required_length - len(y)), mode='constant')

    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, 
                                       hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize from [-80,0] to [0,1]
    S_norm = (S_db + 80.0) / 80.0
    # Add channel dimension for CNN
    S_norm = np.expand_dims(S_norm, axis=-1)  # shape: (128, time_frames, 1)
    return S_norm

def main():
    X_data = []
    y_data = []
    classes = set()

    for note_dir in os.listdir(RAW_DATA_DIR):
        note_path = os.path.join(RAW_DATA_DIR, note_dir)
        if os.path.isdir(note_path):
            label = note_dir
            for f in os.listdir(note_path):
                if f.endswith(".wav"):
                    file_path = os.path.join(note_path, f)
                    mel_spec = extract_mel_spectrogram(file_path)
                    X_data.append(mel_spec)
                    y_data.append(label)
                    classes.add(label)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    classes = sorted(list(classes))

    # Create directories if needed
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)

    np.savez(OUTPUT_NPZ, X=X_data, y=y_data, classes=classes)
    print(f"Saved processed data to {OUTPUT_NPZ}")
    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)
    print("Classes:", classes)

if __name__ == "__main__":
    main()

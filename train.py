#!/usr/bin/env python3
"""
train.py
---------------
End-to-end script to train the CNN model on GuitarSet data,
and splitting data into 80% train, 10% test, 10% validation.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from parse_jams import parse_guitarset_jams
from feature_extraction import extract_cqt_frames
from model_cnn import build_cnn

# Enable GPU
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU:", physical_devices[0])
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPU found; using CPU instead.")

# For standard tuning
STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]
MAX_FRET = 19
MUTED_IDX = 20  # We'll treat 0..19 as actual frets, index 20 => muted

def fret_to_onehot(fret: int) -> np.ndarray:
    """
    Map fret in [-1..19] => index in [0..20].
    -1 => 20 (muted).
    Returns a (21,) one-hot vector.
    """
    n_classes = MAX_FRET + 2  # 19 frets + 1 = 20, plus 1 muted => 21
    arr = np.zeros(n_classes, dtype=np.float32)
    if fret < 0:
        arr[MUTED_IDX] = 1.0
    else:
        arr[fret] = 1.0
    return arr

def main():
    # Paths for .wav and .jams files
    WAV_DIR = "data/raw/wav"   # .wav files
    JAMS_DIR = "data/raw/jams" # .jams files

    sr = 22050
    hop_length = 512

    # We'll gather X, Y across all files
    all_X = []
    all_Y = []

    # Loop over WAV files
    for fname in os.listdir(WAV_DIR):
        # We only want files with "_hex_cln.wav" suffix
        if not fname.endswith("_hex_cln.wav"):
            continue
        wav_path = os.path.join(WAV_DIR, fname)
        
        # Remove "_hex_cln.wav" and replace it with ".jams"
        base = fname.replace("_hex_cln.wav", "")
        jams_file = base + ".jams"
        jams_path = os.path.join(JAMS_DIR, jams_file)
        
        if not os.path.exists(jams_path):
            print(f"No matching JAMS for: {fname}")
            continue

        print("Loading JAMS:", jams_path)
        # Parse label (string_labels shape => (n_frames, 6))
        string_labels = parse_guitarset_jams(jams_path, sr, hop_length)

        # Extract features => X_frames shape => (n_frames, 192, 9, 1)
        X_frames = extract_cqt_frames(wav_path, sr, hop_length)
        n_frames = X_frames.shape[0]

        # Align lengths (some files might not match exactly)
        n_labels = string_labels.shape[0]
        min_len = min(n_frames, n_labels)
        X_frames = X_frames[:min_len]
        string_labels = string_labels[:min_len]

        # Convert label to one-hot => shape (n_frames, 6, 21)
        Y_frames = []
        for i in range(min_len):
            row_6 = string_labels[i]  # e.g. [5, -1, 0, 3, ...]
            row_oh = [fret_to_onehot(f) for f in row_6]
            row_oh = np.array(row_oh, dtype=np.float32)  # shape (6, 21)
            Y_frames.append(row_oh)
        Y_frames = np.array(Y_frames, dtype=np.float32)  # shape (n_frames, 6, 21)

        # Accumulate into master lists
        all_X.append(X_frames)
        all_Y.append(Y_frames)

    # Concatenate all data
    X_all = np.concatenate(all_X, axis=0)  # shape (total_frames, 192, 9, 1)
    Y_all = np.concatenate(all_Y, axis=0)  # shape (total_frames, 6, 21)

    print("Total frames collected:", X_all.shape[0])

    # === 80/10/10 Splits ===
    # 1) First split off 10% test => 90% train+val
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_all, Y_all, test_size=0.1
    )
    # 2) From that 90%, split off 10% for validation => 80% train, 10% val
    # 0.1 of 0.9 => 0.09 => ~10% overall
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=0.1111
    )

    print(f"Train set: {X_train.shape[0]} frames")
    print(f"Val set:   {X_val.shape[0]} frames")
    print(f"Test set:  {X_test.shape[0]} frames")

    # Build the CNN model
    model = build_cnn(
        input_shape=(192, 9, 1),
        num_strings=6,
        num_frets=21  # 19 frets + open + muted => 21 classes
    )
    model.summary()

    # Train for fewer epochs if you want quick tests
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=16,
        batch_size=32
    )

    # Evaluate on VAL
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Evaluate on TEST
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    model.save("model.h5")
    print("Model saved as tabcnn_model.h5")

if __name__ == "__main__":
    main()

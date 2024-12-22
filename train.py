#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

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

def preprocess_and_save_data(wav_dir, jams_dir, output_file, sr=22050, hop_length=512):
    """
    Preprocess data and save to a file for reuse.
    """
    all_X = []
    all_Y = []

    # Loop over WAV files
    for fname in os.listdir(wav_dir):
        if not fname.endswith("_hex_cln.wav"):
            continue
        wav_path = os.path.join(wav_dir, fname)
        
        # Remove "_hex_cln.wav" and replace it with ".jams"
        base = fname.replace("_hex_cln.wav", "")
        jams_file = base + ".jams"
        jams_path = os.path.join(jams_dir, jams_file)

        if not os.path.exists(jams_path):
            print(f"No matching JAMS for: {fname}")
            continue

        print("Processing JAMS:", jams_path)
        string_labels = parse_guitarset_jams(jams_path, sr, hop_length)
        X_frames = extract_cqt_frames(wav_path, sr, hop_length)

        # Align lengths
        n_frames = X_frames.shape[0]
        n_labels = string_labels.shape[0]
        min_len = min(n_frames, n_labels)
        X_frames = X_frames[:min_len]
        string_labels = string_labels[:min_len]

        # Convert labels to one-hot encoding
        Y_frames = []
        for i in range(min_len):
            row_6 = string_labels[i]
            row_oh = [fret_to_onehot(f) for f in row_6]
            Y_frames.append(np.array(row_oh, dtype=np.float32))

        all_X.append(X_frames)
        all_Y.append(np.array(Y_frames, dtype=np.float32))

    # Save processed data
    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    with open(output_file, 'wb') as f:
        pickle.dump((X_all, Y_all), f)
    print(f"Data saved to {output_file}")

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from file.
    """
    with open(file_path, 'rb') as f:
        X_all, Y_all = pickle.load(f)
    return X_all, Y_all

def main():
    # Paths
    WAV_DIR = "data/raw/wav"
    JAMS_DIR = "data/raw/jams"
    DATA_FILE = "preprocessed_data.pkl"
    sr = 22050
    hop_length = 512

    # Preprocess data if not already saved
    if not os.path.exists(DATA_FILE):
        preprocess_and_save_data(WAV_DIR, JAMS_DIR, DATA_FILE, sr, hop_length)

    # Load preprocessed data
        
    X_all, Y_all = load_preprocessed_data(DATA_FILE)
    print(f"Number of samples in dataset: {X_all.shape[0]}")
    print("Total frames loaded:", X_all.shape[0])
    print(f"Input data shape: {X_all.shape}")
    print(f"Labels shape: {Y_all.shape}")

    # === 80/10/10 Splits ===
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_all, Y_all, test_size=0.1
    )
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
        num_frets=21
    )
    model.summary()

    # Add ModelCheckpoint to save model after every epoch
    checkpoint_callback = ModelCheckpoint(
        filepath="checkpoints/model_epoch_{epoch:02d}.keras",
        save_best_only=False,
        save_weights_only=False,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint_callback]
    )

    # Evaluate on VAL
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Evaluate on TEST
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save final model
    model.save("model.h5")
    print("Final model saved as model.h5")

if __name__ == "__main__":
    main()

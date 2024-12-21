#!/usr/bin/env python3

import numpy as np
import librosa
import tensorflow as tf

from feature_extraction import extract_cqt_frames

MAX_FRET = 19
MUTED_IDX = 20  # index for muted
SAMPLE_RATE = 22050
HOP_LENGTH = 512

def load_tabcnn_model(model_path: str):
    """
    Load the trained TabCNN model from disk.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def predict_tablature(audio_path: str, model) -> np.ndarray:
    """
    Generates a (num_frames, 6) array of fret predictions for the given audio file.
    """
    # 1) Extract the same features you used in training
    X_frames = extract_cqt_frames(audio_path, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

    # 2) Model inference => shape (num_frames, 6, 21) in your case
    predictions = model.predict(X_frames, verbose=1)

    pred_frets = np.argmax(predictions, axis=-1)

    return pred_frets

def print_tablature(pred_frets: np.ndarray):
    """
    Simple textual display of a frame-level tab representation.
    pred_frets shape: (num_frames, 6)
    Each row = a time frame, 6 columns = strings [lowest=0 .. highest=5].
    
    This is just a naive text display. More advanced code might group frames
    or combine consecutive frames with the same note.
    """
    num_frames = pred_frets.shape[0]
    for i in range(num_frames):
        frame_label = pred_frets[i]
        string_labels = []
        for fret_idx in frame_label:
            if fret_idx == MUTED_IDX:
                string_labels.append('X')
            else:
                string_labels.append(str(fret_idx))
        print(f"Frame {i}: {string_labels}")

def main():
    model_path = "tabcnn_model.h5"    
    audio_path = "audioconvert/CMajor.m4a" 

    # 1) Load the model
    model = load_tabcnn_model(model_path)
    print("Model loaded.")

    # 2) Predict
    pred_frets = predict_tablature(audio_path, model)
    # shape => (num_frames, 6)

    print("pred_frets shape:", pred_frets.shape)

    # 3) Print or render the tab
    print_tablature(pred_frets)

if __name__ == "__main__":
    main()

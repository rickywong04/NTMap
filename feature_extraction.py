#!/usr/bin/env python3

import os
import numpy as np
import librosa

def extract_cqt_frames(
    wav_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    n_bins: int = 192,
    bins_per_octave: int = 24,
    context_size: int = 9
) -> np.ndarray:
    """
    Load audio, compute CQT, and create context-windowed frames.
    Return shape: (n_frames, freq_bins=192, context_size=9, 1)
    """
    y, sr = librosa.load(wav_path, sr=sr)
    C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)
    C_mag = np.abs(C)  # shape (n_bins, n_frames)
    n_frames = C_mag.shape[1]
    half_ctx = context_size // 2

    # Padding
    pad_left = np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    pad_right = np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    C_padded = np.concatenate([pad_left, C_mag, pad_right], axis=1)

    X_list = []
    for i in range(n_frames):
        segment = C_padded[:, i : i+context_size]  # (192, 9)
        X_list.append(segment)

    X = np.array(X_list)[..., np.newaxis]  # (n_frames, 192, 9, 1)
    return X

def extract_all_audio(
    audio_dir: str,
    sr: int = 22050,
    hop_length: int = 512
):
    """
    Example function to iterate over .wav audio, extract CQT.
    """
    data = {}
    for fname in os.listdir(audio_dir):
        if fname.endswith(".wav"):
            wav_path = os.path.join(audio_dir, fname)
            X = extract_cqt_frames(wav_path, sr, hop_length)
            data[fname] = X
    return data

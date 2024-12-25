#!/usr/bin/env python3
"""
feature_extraction.py
---------------------
Uses librosa to load audio and compute CQT frames.
"""

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
    Return shape: (n_frames, 1, 192, 9) in PyTorch's (N, C, H, W) format.
    """
    y, sr = librosa.load(wav_path, sr=sr)
    C = librosa.cqt(y, sr=sr, hop_length=hop_length,
                    n_bins=n_bins, bins_per_octave=bins_per_octave)
    C_mag = np.abs(C)  # (n_bins, n_frames)
    n_frames = C_mag.shape[1]
    half_ctx = context_size // 2

    # Pad left/right
    pad_left = np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    pad_right= np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    C_padded = np.concatenate([pad_left, C_mag, pad_right], axis=1)

    X_list = []
    for i in range(n_frames):
        segment = C_padded[:, i : i+context_size]  # shape (192, 9)
        # We'll store as (1, 192, 9) => channel-first for PyTorch
        segment = segment[None, :, :]  # shape => (1, 192, 9)
        X_list.append(segment)

    X = np.array(X_list, dtype=np.float32)  # shape => (n_frames, 1, 192, 9)
    return X

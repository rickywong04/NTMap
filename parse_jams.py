#!/usr/bin/env python3
"""
parse_jams.py
-------------
"""

import os
import jams
import numpy as np

# For standard-tuning (E2..E4) in MIDI
STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
MAX_FRET = 19  # Up to 19 frets (or 20 if you prefer)

def parse_guitarset_jams(
    jams_path: str,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Parse GuitarSet .jams file. Return a NumPy array of shape (n_frames, 6),
    each entry an integer [0..19] or -1 for muted (no pitch).
    """
    jam = jams.load(jams_path)
    duration_s = jam.file_metadata.duration
    n_frames = int(np.ceil(duration_s * sr / hop_length))

    # Initialize: -1 means string is not sounding
    string_labels = -1 * np.ones((n_frames, 6), dtype=int)

    # Find all pitch_contour annotations:
    all_contours = jam.search(namespace="pitch_contour")
    for ann in jam.annotations:
        if ann.namespace != "pitch_contour":
            continue
        ds = ann.annotation_metadata.data_source
        if ds is None:
            # skip
            continue
        try:
            string_idx = int(ds)  # 0..5
        except:
            # skip if not parseable
            continue
        if not (0 <= string_idx < 6):
            continue

        # Parse the pitch/time from ann.data
        for obs in ann.data:
            t_sec = obs.time
            freq = obs.value
            if isinstance(freq, dict) and "frequency" in freq:
                freq = freq["frequency"]
            if freq <= 0:
                continue

            frame_idx = int(np.round(t_sec * sr / hop_length))
            if not (0 <= frame_idx < n_frames):
                continue

            # Convert freq to MIDI
            midi_val = 69 + 12 * np.log2(freq / 440.0)
            fret = int(round(midi_val)) - STRING_OPEN_MIDI[string_idx]

            # If fret is out of range => -1 means unplayed
            if fret < 0 or fret > MAX_FRET:
                fret = -1

            # Store
            # If multiple partial observations for same frame => keep the latest or ignore
            string_labels[frame_idx, string_idx] = fret

    return string_labels

def parse_all_jams(
    jams_dir: str,
    sr: int = 22050,
    hop_length: int = 512
):
    """
    Parse all .jams files in a directory, returns a dict {filename: string_labels}.
    """
    result = {}
    for fname in os.listdir(jams_dir):
        if fname.endswith(".jams"):
            jams_path = os.path.join(jams_dir, fname)
            labels = parse_guitarset_jams(jams_path, sr, hop_length)
            result[fname] = labels
    return result

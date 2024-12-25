#!/usr/bin/env python3
"""
parse_jams.py
-------------
Same logic as your original TF code: loads .jams, returns shape (n_frames, 6)
with frets in [-1..19], or -1 for muted.
"""

import jams
import numpy as np
import os

# Standard-tuning string MIDI notes
STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2..E4
MAX_FRET = 19

def parse_guitarset_jams(
    jams_path: str,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    jam = jams.load(jams_path)
    duration_s = jam.file_metadata.duration
    n_frames = int(np.ceil(duration_s * sr / hop_length))

    # Initialize => -1 => means not sounding
    string_labels = -1 * np.ones((n_frames, 6), dtype=int)

    # Search pitch_contour in jam
    for ann in jam.annotations:
        if ann.namespace != "pitch_contour":
            continue
        ds = ann.annotation_metadata.data_source
        if ds is None:
            continue
        try:
            string_idx = int(ds)  # 0..5
        except:
            continue
        if not (0 <= string_idx < 6):
            continue

        # parse the obs
        for obs in ann.data:
            t_sec = obs.time
            freq = obs.value
            if isinstance(freq, dict) and "frequency" in freq:
                freq = freq["frequency"]
            if freq <= 0:
                continue

            frame_idx = int(np.round(t_sec * sr / hop_length))
            if frame_idx < 0 or frame_idx >= n_frames:
                continue

            # freq => MIDI => fret
            midi_val = 69 + 12 * np.log2(freq/440.0)
            fret = int(round(midi_val)) - STRING_OPEN_MIDI[string_idx]
            if fret < 0 or fret > MAX_FRET:
                fret = -1
            string_labels[frame_idx, string_idx] = fret

    return string_labels

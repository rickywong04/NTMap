#!/usr/bin/env python3
"""
generate_tabs.py

Generates:
  1) A streamlined ASCII guitar tab from a guitar track audio file.
  2) A MIDI file containing the detected notes, so you can listen
     to the approximate pitch content. 
  3) (Optional) Renders the MIDI to audio if fluidsynth is available
     and a SoundFont file is provided.

Usage Example:
  python generate_tabs.py \
      --input "my_guitar.wav" \
      --model_pt "final_model.pt" \
      --tab_txt "output_tab.txt" \
      --midi_out "detected_notes.mid" \
      --soundfont "my_soundfont.sf2" \
      --audio_out "rendered_from_midi.wav" \
      --chunk_size 16

Prereqs:
  pip install torch librosa soundfile numpy mido scipy
  Also install fluidsynth command line if you want to do MIDI -> audio automatically.
"""

import os
import sys
import argparse
import logging
import subprocess

import numpy as np
import torch
import librosa
import mido

#############################################
# Constants
#############################################

STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2..e4 (Low E to High e)
MAX_FRET = 19
MUTED_IDX = 20
SR = 22050
HOP_LENGTH = 512

#############################################
# Frame Extraction
#############################################

def extract_cqt_frames_simple(wav_path: str):
    sr = SR
    hop_length = HOP_LENGTH
    n_bins = 192
    bins_per_octave = 24
    context_size = 9

    y, _ = librosa.load(wav_path, sr=sr)
    C = librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        n_bins=n_bins, bins_per_octave=bins_per_octave
    )
    C_mag = np.abs(C)  # (192, frames)
    n_frames = C_mag.shape[1]
    half_ctx = context_size // 2

    pad_left  = np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    pad_right = np.zeros((n_bins, half_ctx), dtype=C_mag.dtype)
    C_padded  = np.concatenate([pad_left, C_mag, pad_right], axis=1)

    X_list = []
    for i in range(n_frames):
        segment = C_padded[:, i:i+context_size]  # (192, 9)
        segment = segment[None, :, :]            # (1, 192, 9)
        X_list.append(segment)

    X = np.array(X_list, dtype=np.float32)  # (N, 1, 192, 9)
    return X

#############################################
# ASCII Tab Generation Without Smoothing
#############################################

def ascii_tab_from_frets(
    pred_frets: np.ndarray,
    frame_duration: float = 0.0232,
    chunk_size: int = 8
):
    """
    Generates ASCII tablature by mapping each frame's fret predictions directly
    to the tab without any smoothing or merging.

    Args:
      pred_frets (np.ndarray): shape (N, 6), raw fret predictions for each frame.
      frame_duration (float): duration in seconds of each frame.
      chunk_size (int): number of fret changes before inserting a bar '|'.

    Returns:
      str: multi-line ASCII tablature string.
    """
    # String names in standard tuning from high e to low E
    strings = ["e", "B", "G", "D", "A", "E"]  # Highest to lowest strings

    # Initialize lines for each string
    lines = [s + "|" for s in strings]  # e|, B|, ..., E|

    # Process each frame's fret predictions
    for idx, chord_tuple in enumerate(pred_frets):
        column = []
        # **Reverse the chord_tuple to map high e first**
        reversed_chord = chord_tuple[::-1]  # Now high e is first
        for fret in reversed_chord:
            if fret == MUTED_IDX:
                fret_str = "X"
            elif fret == 0:
                fret_str = "0"
            elif fret < 10:
                fret_str = f"{fret}"
            else:
                fret_str = f"({fret})"
            column.append(fret_str)
        
        # Insert bar every 'chunk_size' columns
        if idx > 0 and (idx % chunk_size == 0):
            for i in range(6):
                lines[i] += "|"
        
        # Append fret symbols to each string line
        for i in range(6):
            lines[i] += column[i] + "-"
    
    # Close each line with a bar
    for i in range(6):
        lines[i] += "|"

    # Combine all lines into a single string
    ascii_tab = "\n".join(lines)
    return ascii_tab

#############################################
# MIDI Generation Without Smoothing
#############################################

def fret_to_midi_note(string_idx, fret):
    """
    Converts (string_idx, fret) to MIDI note number.
    Returns None if the fret is muted or invalid.

    Args:
      string_idx (int): Index of the string (0-5 for e-B-G-D-A-E).
      fret (int): Fret number.

    Returns:
      int or None: MIDI note number or None.
    """
    if fret == MUTED_IDX:
        return None
    if fret < 0 or fret > MAX_FRET:
        return None
    base_note = STRING_OPEN_MIDI[string_idx]
    return base_note + fret

def create_midi_from_frets(
    pred_frets: np.ndarray,
    frame_duration=0.0232
):
    """
    Builds a MIDI object from raw fret predictions without any merging.

    Args:
      pred_frets (np.ndarray): shape (N, 6) array of raw fret predictions.
      frame_duration (float): time in seconds for each frame.

    Returns:
      mido.MidiFile: Generated MIDI file object.
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage

    midi_out = MidiFile()
    track = MidiTrack()
    midi_out.tracks.append(track)

    tempo_bpm = 120
    us_per_beat = int(60_000_000 // tempo_bpm)
    track.append(MetaMessage('set_tempo', tempo=us_per_beat, time=0))

    ticks_per_beat = midi_out.ticks_per_beat

    def seconds_to_ticks(sec):
        beats = sec * (tempo_bpm / 60.0)
        return int(beats * ticks_per_beat)

    n_frames = pred_frets.shape[0]

    # Keep track of currently active notes to handle note_off
    active_notes = {}

    for f in range(n_frames):
        chord = pred_frets[f]
        current_time_ticks = seconds_to_ticks(f * frame_duration) if f == 0 else 0

        # Determine notes to turn on and off
        notes_to_turn_on = []
        notes_to_turn_off = []

        for s_idx, fret in enumerate(chord):
            midi_note = fret_to_midi_note(s_idx, fret)
            if midi_note is not None:
                if midi_note not in active_notes:
                    notes_to_turn_on.append(midi_note)
                    active_notes[midi_note] = f  # Track when it was turned on
            else:
                # If fret is muted, turn off any active note on this string
                base_note = STRING_OPEN_MIDI[s_idx]
                for note in list(active_notes):
                    if base_note <= note <= base_note + MAX_FRET:
                        notes_to_turn_off.append(note)

        # Turn off notes first
        for note in notes_to_turn_off:
            if note in active_notes:
                delta_ticks = seconds_to_ticks((f - active_notes[note]) * frame_duration)
                track.append(Message('note_off', note=note, velocity=0, time=delta_ticks))
                del active_notes[note]

        # Then turn on new notes
        for note in notes_to_turn_on:
            track.append(Message('note_on', note=note, velocity=80, time=current_time_ticks))
            active_notes[note] = f

    # Turn off any remaining active notes at the end
    final_time_ticks = seconds_to_ticks(n_frames * frame_duration)
    for note in list(active_notes):
        delta_ticks = final_time_ticks - seconds_to_ticks(active_notes[note] * frame_duration)
        track.append(Message('note_off', note=note, velocity=0, time=delta_ticks))
        del active_notes[note]

    return midi_out

#############################################
# MIDI Rendering with FluidSynth
#############################################

def render_midi_to_audio(midi_path: str, sf2_path: str, audio_out: str):
    """
    Renders MIDI to audio using FluidSynth.

    Args:
      midi_path (str): Path to the MIDI file.
      sf2_path (str): Path to the SoundFont file (.sf2).
      audio_out (str): Path to the output audio file (.wav).

    Note:
      - Ensure FluidSynth is installed and accessible in your system's PATH.
    """
    cmd = [
        "fluidsynth",
        "-ni",
        sf2_path,
        midi_path,
        "-F",
        audio_out,
        "-r",
        "44100"
    ]
    logging.info(f"Rendering MIDI to audio with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"Rendered audio saved to {audio_out}")
    except Exception as e:
        logging.warning(f"Failed to render audio via FluidSynth: {e}")

#############################################
# Model Loading
#############################################

def load_tab_model(model_path: str, device: torch.device):
    from model_cnn import build_cnn

    logging.info(f"Loading PyTorch model from {model_path}")
    model = build_cnn(num_strings=6, num_frets=21)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        logging.error(f"Failed to load model state_dict: {e}")
        sys.exit(1)
    model.to(device)
    model.eval()
    return model

#############################################
# Main Pipeline
#############################################

def main():
    parser = argparse.ArgumentParser(
        description="Generate ASCII tabs & optional MIDI from a guitar audio file."
    )
    parser.add_argument("--input", required=True, help="Path to the guitar track audio file.")
    parser.add_argument("--model_pt", default="final_model.pt", help="Path to the .pt model.")
    parser.add_argument("--tab_txt", default=None, help="Path to output ASCII tab.")
    parser.add_argument("--midi_out", default=None, help="Path to output MIDI file. If not given, no MIDI is saved.")
    parser.add_argument("--soundfont", default=None, help="Path to .sf2 SoundFont for rendering MIDI -> audio.")
    parser.add_argument("--audio_out", default=None, help="Path to output rendered audio file from the MIDI + SoundFont.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Number of fret changes before a bar in ASCII tab.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_tab_model(args.model_pt, device=device)

    # Extract frames
    logging.info(f"Extracting CQT frames from {args.input}")
    X = extract_cqt_frames_simple(args.input)
    X_torch = torch.from_numpy(X).float().to(device)

    # Predict fret positions
    logging.info("Running model inference.")
    with torch.no_grad():
        outputs = model(X_torch)  # shape (N,6,21)

    pred_frets = outputs.argmax(dim=-1).cpu().numpy()

    # ASCII tab
    logging.info("Generating ASCII tab.")
    ascii_tab = ascii_tab_from_frets(pred_frets, chunk_size=args.chunk_size)

    if args.tab_txt:
        try:
            with open(args.tab_txt, "w") as f:
                f.write(ascii_tab)
            logging.info(f"Saved ASCII tab to {args.tab_txt}")
        except Exception as e:
            logging.error(f"Failed to write ASCII tab: {e}")
            sys.exit(1)
    else:
        print("===== ASCII TAB =====")
        print(ascii_tab)
        print("=====================")

    # MIDI export
    if args.midi_out:
        frame_dur_s = float(HOP_LENGTH) / float(SR)  # ~0.0232 seconds per frame
        mid_obj = create_midi_from_frets(pred_frets, frame_duration=frame_dur_s)
        try:
            mid_obj.save(args.midi_out)
            logging.info(f"MIDI saved to {args.midi_out}")
        except Exception as e:
            logging.error(f"Failed to save MIDI: {e}")
            sys.exit(1)

        # Optionally, render the MIDI -> audio with FluidSynth
        if args.soundfont and args.audio_out:
            render_midi_to_audio(args.midi_out, args.soundfont, args.audio_out)
        else:
            if args.audio_out:
                logging.warning("audio_out specified, but no soundfont provided.")
            if args.soundfont:
                logging.warning("soundfont provided, but no audio_out specified.")

if __name__ == "__main__":
    main()

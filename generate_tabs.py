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
      --guitar_wav "my_guitar.wav" \
      --model_pt "final_model.pt" \
      --tab_txt "output_tab.txt" \
      --midi_out "detected_notes.mid" \
      --soundfont "my_soundfont.sf2" \
      --audio_out "rendered_from_midi.wav" \
      --max_jump 4 \
      --chunk_size 16

Prereqs:
  pip install torch librosa soundfile numpy mido
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
# We'll reuse the code from your script:
#############################################

STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2..E4
MAX_FRET = 19
MUTED_IDX = 20
SR = 22050
HOP_LENGTH = 512

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

def smooth_frets(pred_frets: np.ndarray, max_jump=4):
    out = pred_frets.copy()
    n_frames, n_strings = out.shape

    for f in range(1, n_frames):
        for s in range(n_strings):
            prev_fret = out[f-1, s]
            curr_fret = out[f, s]
            if prev_fret == MUTED_IDX or curr_fret == MUTED_IDX:
                continue
            diff = abs(curr_fret - prev_fret)
            if diff > max_jump:
                sign = 1 if curr_fret > prev_fret else -1
                new_fret = prev_fret + sign * max_jump
                new_fret = max(0, min(new_fret, MAX_FRET))
                out[f, s] = new_fret
    return out

def ascii_tab_from_frets(
    pred_frets: np.ndarray,
    frame_duration: float = 0.05,
    min_chord_dur: float = 0.3,
    chunk_size: int = 8
):
    """
    Creates a significantly simplified ASCII tab by merging short chord changes
    and displaying columns only when the chord changes. This drastically
    reduces 'flickering' or noise in the tab.

    Args:
      pred_frets (np.ndarray): shape (N, 6), fret predictions for each frame.
      frame_duration (float): duration in seconds of each frame.
      min_chord_dur (float): minimum chord duration in seconds; shorter chords get merged/skipped.
      chunk_size (int): columns per chunk before inserting a bar '|'.

    Returns:
      str: multi-line ASCII tablature string.
    """
    strings = ["e", "B", "G", "D", "A", "E"]  # top to bottom in final display
    # We'll build chord events: (start_sec, end_sec, chord_tuple).
    chord_events = []
    N = pred_frets.shape[0]
    last_chord = None
    last_start_sec = 0.0

    # 1) gather chord events from frame-level changes
    for i in range(N):
        chord = tuple(pred_frets[i].tolist())  # (6,) as a tuple
        if last_chord is None:
            last_chord = chord
            last_start_sec = i * frame_duration
        else:
            if chord != last_chord:
                end_sec = i * frame_duration
                chord_events.append((last_start_sec, end_sec, last_chord))
                last_chord = chord
                last_start_sec = end_sec

    # add final chord if any
    if last_chord is not None:
        chord_events.append((last_start_sec, N * frame_duration, last_chord))

    # 2) merge or remove short chord events
    merged_events = []
    i = 0
    while i < len(chord_events):
        start_sec, end_sec, chord = chord_events[i]
        dur = end_sec - start_sec
        if dur >= min_chord_dur:
            # keep
            merged_events.append((start_sec, end_sec, chord))
            i += 1
        else:
            # short chord => try merging with prev or next if same chord
            chord_before = merged_events[-1] if merged_events else None
            chord_after = chord_events[i+1] if (i+1 < len(chord_events)) else None
            merged = False
            if chord_before is not None:
                (pstart, pend, pchord) = chord_before
                if pchord == chord:
                    # merge with previous chord
                    merged_events[-1] = (pstart, end_sec, pchord)
                    merged = True
            if (not merged) and chord_after is not None:
                (nstart, nend, nchord) = chord_after
                if nchord == chord:
                    # merge with next chord
                    chord_events[i+1] = (start_sec, nend, nchord)
                    merged = True
            # if not merged => skip
            i += 1

    # 3) second pass: sort & unify identical adjacent chords
    merged_events_sorted = sorted(merged_events, key=lambda x: x[0])
    final_events = []
    for evt in merged_events_sorted:
        if not final_events:
            final_events.append(evt)
        else:
            (prev_start, prev_end, prev_chord) = final_events[-1]
            (cur_start, cur_end, cur_chord) = evt
            # fix overlap
            if cur_start < prev_end:
                cur_start = prev_end
            if prev_chord == cur_chord:
                # unify
                final_events[-1] = (prev_start, cur_end, prev_chord)
            else:
                # append
                final_events.append((cur_start, cur_end, cur_chord))

    # 4) Build ASCII tab from final events
    # We'll create one "column" (fret pattern) per chord event
    lines = [s + "|" for s in strings]  # each line starts w/ e|, B|, ...
    tab_columns = []
    for (start_sec, end_sec, chord_tuple) in final_events:
        # build a column of 6 strings
        column = []
        for s_idx, fret in enumerate(chord_tuple):
            if fret == 20:
                fret_str = "X"
            elif fret == 0:
                fret_str = "0"
            elif fret < 0:
                fret_str = ""  # negative or not recognized => skip
            elif fret < 10:
                fret_str = f"{fret}"
            else:
                fret_str = f"({fret})"
            column.append(fret_str)
        tab_columns.append(column)

    # We only add columns if we have them; chunk them
    for idx, column in enumerate(tab_columns):
        # insert bar line every chunk_size
        if idx > 0 and (idx % chunk_size == 0):
            for i in range(6):
                lines[i] += "|"
        # add the fret markers
        for i in range(6):
            fret_str = column[i]
            # dash after fret
            lines[i] += fret_str + "-"

    # close each line
    for i in range(6):
        lines[i] += "|"

    return "\n".join(lines)


#############################################
# 1) Convert fret predictions to MIDI notes
#############################################

def fret_to_midi_note(string_idx, fret):
    """
    Convert (string_idx, fret) -> MIDI note number.
    If fret == MUTED_IDX or invalid => return None
    """
    if fret == MUTED_IDX:
        return None
    if fret < 0 or fret > MAX_FRET:
        return None
    base_note = STRING_OPEN_MIDI[string_idx]
    return base_note + fret

def create_midi_from_frets(
    pred_frets: np.ndarray,
    frame_duration=0.05,
    min_chord_dur=0.3
):
    """
    Build a MIDI object from the predicted frets, merging consecutive
    frames and also filtering out very short chord events to reduce 'noise'.

    Args:
      pred_frets: shape (N, 6) array of fret predictions.
      frame_duration: time in seconds for each frame.
      min_chord_dur: minimum chord duration in seconds. 
                     short chords under this will be merged/skipped.
    Returns:
      mido.MidiFile object
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    
    # 1) Build raw chord events from frame changes
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
    chord_events = []  # list of (start_sec, end_sec, chord)
    last_chord = None
    last_start_time = 0.0

    # gather chord events
    for f in range(n_frames):
        chord = tuple(pred_frets[f].tolist())  # shape (6,)
        if last_chord is None:
            last_chord = chord
            last_start_time = f * frame_duration
        else:
            if chord == last_chord:
                # same chord => continue
                pass
            else:
                end_time = f * frame_duration
                chord_events.append((last_start_time, end_time, last_chord))
                last_chord = chord
                last_start_time = end_time
    # add the last chord
    if last_chord is not None:
        chord_events.append((last_start_time, n_frames*frame_duration, last_chord))

    # 2) Merge or remove short chord events under min_chord_dur
    #    We'll do a simple pass that if a chord is < min_chord_dur:
    #    - if it matches the chord before or after, we merge with that chord.
    #    - otherwise we skip it.
    merged_events = []
    i = 0
    while i < len(chord_events):
        start_sec, end_sec, chord = chord_events[i]
        dur = end_sec - start_sec
        if dur >= min_chord_dur:
            # keep it
            merged_events.append((start_sec, end_sec, chord))
            i += 1
        else:
            # short chord => see if we can merge it with prev or next
            chord_before = merged_events[-1] if merged_events else None
            chord_after = chord_events[i+1] if (i+1 < len(chord_events)) else None

            # if same as chord_before, extend chord_before's end_sec
            # or if same as chord_after, shift chord_after's start_sec
            merged = False
            if chord_before is not None:
                (pstart, pend, pchord) = chord_before
                if pchord == chord:
                    # merge: extend previous chord
                    merged_events[-1] = (pstart, end_sec, pchord)
                    merged = True
            if (not merged) and chord_after is not None:
                (nstart, nend, nchord) = chord_after
                if nchord == chord:
                    # merge: shift next chord's start
                    chord_events[i+1] = (start_sec, nend, nchord)
                    merged = True
            # if we didn't merge, we skip
            if not merged:
                # do nothing => skip
                pass
            i += 1

    # 3) Possibly do a second pass: if merging caused overlaps or out-of-order times
    #    let's just do a quick pass to re-sort, and merge again if needed
    merged_events_sorted = sorted(merged_events, key=lambda x: x[0])

    # second pass: if consecutive identical chords or overlap => merge them
    final_events = []
    for evt in merged_events_sorted:
        if not final_events:
            final_events.append(evt)
        else:
            (prev_start, prev_end, prev_chord) = final_events[-1]
            (cur_start, cur_end, cur_chord) = evt
            if cur_start < prev_end:
                # overlap => adjust
                cur_start = prev_end
            if prev_chord == cur_chord:
                # merge them
                final_events[-1] = (prev_start, cur_end, prev_chord)
            else:
                final_events.append((cur_start, cur_end, cur_chord))

    # 4) Build actual MIDI note_on/note_off from final events
    current_time_ticks = 0
    for (start_sec, end_sec, chord) in final_events:
        start_ticks = seconds_to_ticks(start_sec)
        end_ticks   = seconds_to_ticks(end_sec)
        dur_ticks   = end_ticks - start_ticks
        dt = max(0, start_ticks - current_time_ticks)

        # turn chord on
        notes_on = []
        for s_idx, fret in enumerate(chord):
            midi_note = fret_to_midi_note(s_idx, fret)
            if midi_note is not None:
                track.append(Message('note_on', note=midi_note, velocity=80, time=dt))
                dt = 0
                notes_on.append(midi_note)

        # note_off after dur_ticks
        dt_off = max(0, dur_ticks)
        for i, note_val in enumerate(notes_on):
            if i == 0:
                track.append(Message('note_off', note=note_val, velocity=0, time=dt_off))
            else:
                track.append(Message('note_off', note=note_val, velocity=0, time=0))

        current_time_ticks = end_ticks

    return midi_out

def render_midi_to_audio(midi_path: str, sf2_path: str, audio_out: str):
    """
    Calls `fluidsynth` externally to render the .mid -> .wav 
    using the given SoundFont .sf2 file.
    Make sure 'fluidsynth' is installed.
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
    except Exception as e:
        logging.warning(f"Failed to render audio via fluidsynth: {e}")

#############################################
# Model loading, main pipeline
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

def main():
    parser = argparse.ArgumentParser(
        description="Generate ASCII tabs & optional MIDI from a guitar audio file."
    )
    parser.add_argument("--guitar_wav", required=True, help="Path to the guitar track audio file.")
    parser.add_argument("--model_pt", default="final_model.pt", help="Path to the .pt model.")
    parser.add_argument("--tab_txt", default=None, help="Path to output ASCII tab.")
    parser.add_argument("--midi_out", default=None, help="Path to output MIDI file. If not given, no MIDI is saved.")
    parser.add_argument("--soundfont", default=None, help="Path to .sf2 SoundFont for rendering MIDI -> audio.")
    parser.add_argument("--audio_out", default=None, help="Path to output rendered audio file from the MIDI + SoundFont.")
    parser.add_argument("--max_jump", type=int, default=4, help="Max fret jump for smoothing.")
    parser.add_argument("--chunk_size", type=int, default=16, help="Number of tab changes before a bar in ASCII tab.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model = load_tab_model(args.model_pt, device=device)

    # Extract frames
    logging.info(f"Extracting CQT frames from {args.guitar_wav}")
    X = extract_cqt_frames_simple(args.guitar_wav)
    X_torch = torch.from_numpy(X).float().to(device)

    # Predict fret positions
    logging.info("Running model inference.")
    with torch.no_grad():
        outputs = model(X_torch)  # shape (N,6,21)

    pred_frets = outputs.argmax(dim=-1).cpu().numpy()

    # Smooth
    logging.info("Smoothing frets.")
    smoothed_frets = smooth_frets(pred_frets, max_jump=args.max_jump)

    # ASCII tab
    logging.info("Generating ASCII tab.")
    ascii_tab = ascii_tab_from_frets(smoothed_frets, chunk_size=args.chunk_size)

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
        frame_dur_s = float(HOP_LENGTH) / float(SR)  # ~0.0232
        mid_obj = create_midi_from_frets(smoothed_frets, frame_duration=frame_dur_s)
        try:
            mid_obj.save(args.midi_out)
            logging.info(f"MIDI saved to {args.midi_out}")
        except Exception as e:
            logging.error(f"Failed to save MIDI: {e}")
            sys.exit(1)

        # Optionally, render the MIDI -> audio with fluidsynth
        if args.soundfont and args.audio_out:
            render_midi_to_audio(args.midi_out, args.soundfont, args.audio_out)
        else:
            if args.audio_out:
                logging.warning("audio_out specified, but no soundfont provided.")
            if args.soundfont:
                logging.warning("soundfont provided, but no audio_out specified.")

if __name__ == "__main__":
    main()

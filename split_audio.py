#!/usr/bin/env python3
"""
split_audio.py

A minimal script that uses audio-separator to separate an audio file,
unless it looks like it's already a guitar track.

Usage:
  python split_audio.py --input_audio "mySong.mp3" \
                        --output_dir "out" \
                        --model_filename "htdemucs_6s.yaml"

Installation/Prereqs:
  pip install audio-separator

The script prints the path to the guitar track once done.
"""

import os
import sys
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Check if input file is guitar track; if not, run audio-separator and return the guitar track path."
    )
    parser.add_argument("--input_audio", required=True, help="Path to the input audio file.")
    parser.add_argument("--output_dir", default="separated_out", help="Directory for separated stems (default: 'separated_out').")
    parser.add_argument("--model_filename", default="htdemucs_6s.yaml",
                        help="audio-separator model. (default: 'htdemucs_6s.yaml')")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    input_path = args.input_audio
    base_lower = os.path.basename(input_path).lower()

    # If name suggests it's already a guitar track, skip separation
    if ("guitar" in base_lower) or ("demucs" in base_lower):
        logging.info(f"Detected guitar/demucs in filename => skipping separation.")
        print(input_path)  # Print the path as "this is the guitar track"
        return

    # Otherwise, run audio-separator
    logging.info("Running audio-separator to separate the track.")
    from audio_separator.separator import Separator

    sep = Separator(
        output_dir=args.output_dir,
        output_format="WAV",  # safer format
        log_level=logging.INFO
    )
    sep.load_model(model_filename=args.model_filename)
    out_paths = sep.separate(input_path)

    # Find guitar track
    guitar_path = None
    for p in out_paths:
        pname = os.path.basename(p).lower()
        if "guitar" in pname:
            guitar_path = p
            break

    # If not found, fallback to "other" or "instrument"
    if not guitar_path:
        for p in out_paths:
            pname = os.path.basename(p).lower()
            if "other" in pname or "instrument" in pname:
                guitar_path = p
                break

    if not guitar_path and len(out_paths) > 0:
        guitar_path = out_paths[0]

    if not guitar_path:
        logging.error("No guitar or fallback track found after separation.")
        sys.exit(1)

    logging.info(f"Guitar track => {guitar_path}")
    print(guitar_path)


if __name__ == "__main__":
    main()

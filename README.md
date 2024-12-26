**NTMap: Neural Tone Mapping and Processing**

**Overview**

NTMap is a machine learning project that uses Convolutional Neural Networks (CNNs) to classify polyphonic acoustic guitar notes and generate corresponding guitar tablature. This project uses a neural network model, to analyze audio frames to predict fret positions on a guitar, generating both text-based tabs and MIDI representations of the performance. Additionally, it offers the option to split audio into separate stems to isolate a guitar track.

**Dataset Information (GuitarSet)**

GuitarSet was used to train this model to detect multiple notes across the fretboard. This dataset came with 360 excerpts of acoustic guitar audio recorded with a hexaphonic pickup that records the input audio per string. 

More information can be found here (https://zenodo.org/records/3371780)


**Features**

- **ASCII Tablature Generation:** Converts guitar audio into guitar tabs.
- **Audio Splitting:** Convert audio into separate stems (i.e. guitar, bass, keys, etc.) 
- **MIDI File Creation:** Generates MIDI files from fret predictions
- **Frame-by-Frame:** Uses the model's predictions to capture all frame-by-frame fret changes.

**Prerequisites**

Before setting up the project, ensure you have the following installed:

- **Python 3.7 or higher**
- **Pip** (Python package installer)

**Python Libraries**

Install the required Python libraries using pip:

```pip install torch torchaudio audio-separator[cpu] librosa soundfile numpy matplotlib mido scipy ```

Alternatively, you can also install ```pip install audio-separator[gpu]``` if your device is compatabile. More information about that can be found here: https://github.com/nomadkaraoke/python-audio-separator

**Installation**

**Clone the Repository:**

``` git clone https://github.com/rickywong04/Neural-Tone-Mapping-and-Processing.git ```

**Usage**

```generate_tabs.py``` processes a guitar audio file to generate ASCII tabs and MIDI files.

**Command-Line Arguments**

- --input *(required)*: Path to the input guitar audio file (e.g., my\_guitar.wav).
- --model\_pt *(optional)*: Path to the trained PyTorch model file (default: final\_model.pt).
- --tab\_txt *(optional)*: Path to save the generated ASCII tablature (e.g., output\_tab.txt). If not provided, the tab will be printed to the console.
- --midi\_out *(optional)*: Path to save the generated MIDI file (e.g., detected\_notes.mid). If not provided, no MIDI file will be created.
- --soundfont *(optional)*: Path to a .sf2 SoundFont file for rendering MIDI to audio (e.g., my\_soundfont.sf2). Required if --audio\_out is specified.
- --audio\_out *(optional)*: Path to save the rendered audio file from the MIDI (e.g., rendered\_from\_midi.wav). Requires both --midi\_out and --soundfont.
- --chunk\_size *(optional)*: Number of fret changes before inserting a bar | in the ASCII tab for readability (default: 16).

**Example Usage**

**Generating ASCII Tab Only:** 



```bash
python generate_tabs.py 
--input "my_guitar.wav" --model_pt "final_model.pt" --tab_txt "output_tab.txt" --chunk_size 16
```

*If --tab\_txt is not provided, the ASCII tab will be displayed in the console.*

**Generating ASCII Tab and MIDI File:** 

```bash
python generate_tabs.py 
--input "my_guitar.wav" --model_pt "final_model.pt" --tab_txt "output_tab.txt" 
--midi_out "detected_notes.mid" \ --chunk_size 16
```

**Generating ASCII Tab, MIDI File, and Rendering Audio:** 


```bash
python generate_tabs.py 
--input "my_guitar.wav" --model_pt "final_model.pt" --tab_txt "output_tab.txt" 
--midi_out "detected_notes.mid" --soundfont "my_soundfont.sf2"  --chunk_size 16
```
**Audio Splitting**
```split_audio.py```
The split_audio.py script uses audio-separator to separate an audio file into different stems. If the input audio file already appears to be a guitar track based on its filename, the script skips the separation process.

**Usage**
To use the script, run the following command:


```bash
python split_audio.py --input_audio "mySong.mp3" --output_dir "out" --model_filename "htdemucs_6s.yaml"
```
Command-Line Arguments
--input_audio (required): Path to the input audio file (e.g., mySong.mp3).
--output_dir (optional): Directory where the separated stems will be saved (default: separated_out).
--model_filename (optional): Filename of the audio-separator model to use (default: htdemucs_6s.yaml).

**How It Works**

1. **Audio Processing:**
- The script loads the input guitar audio file using librosa.
- It computes the Constant-Q Transform (CQT) of the audio to extract frequency-domain features.
- Frames are extracted with a specified context size to capture temporal information.
2. **Fret Prediction:**
- The processed frames are fed into a pre-trained neural network model built with PyTorch.
- The model predicts fret positions for each of the six guitar strings across all frames.
- Each frame's fret predictions are mapped to their corresponding guitar strings.




**Exmaple ASCII Tablature (output\_tab.txt):**

A text file containing the generated guitar tablature in ASCII format. 


```
===== ASCII TAB =====
e|0-0-2-2-3-3-2-0-|
B|1-1-3-3-4-4-3-1-|
G|0-0-2-2-4-4-2-0-|
D|2-2-4-4-5-5-4-2-|
A|3-3-5-5-7-7-5-3-|
E|X-X-0-0-0-0-0-X-|
=====================
```

**Training**
If you want to train your own model, please download the GuitarSet ```audio_hex-pickup_debleeded.zip``` along with ```annotations.zip```. The audio files should be placed in ```data/raw/wav``` and the annotations (*.jams files) should be placed in ```data/raw/jams```. 

**Contributing**

Contributions are welcome! If you have suggestions, bug reports, or improvements, feel free to open an issue or submit a pull request.


## Acknowledgements

This is a proof of concept so far! This currently stil is not completely accurate and will get thrown off by random noise. However, I plan to keep improving and add more implementation to this project.
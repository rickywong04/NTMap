
# NTMap: Neural Tone Mapping and Processing

 

NTMap is a machine learning project that utilizes Convolutional Neural Networks (CNNs) to classify acoustic guitar notes and generate corresponding guitar tablature. The dataset was taken from 'https://www.kaggle.com/datasets/mohammedalkooheji/guitar-notes-dataset/data', and it consists of pre-recorded audio samples of every note on a standard 6-string guitar.

  

## Features (Note: This is still a work in progress!)

-  **Note Classification:** Identify notes up to the from E2 to E6 (range of a 24-fret guitar).

-  **Guitar Tablature Generation:** Convert predicted notes into guitar string and fret positions.

-  **CNN Architecture:** Uses to process Mel-spectrograms of the audio.

  

---

  

## How It Works

  

### 1. Preprocessing the Dataset

The `preprocess.py` script converts raw audio data into normalized Mel-spectrograms:

-  **Input:**  `.wav` audio files (2 seconds, 44.1 kHz, mono).

-  **Output:**  `processed_data.npz`, a compressed file containing:

-  `X`: Mel-spectrograms as 2D image-like arrays.

-  `y`: Corresponding note labels.

-  `classes`: All unique note classes in the dataset.

  

### 2. Training the CNN

The `train_model.py` script trains a CNN to classify notes:

-  **Input:**  `processed_data.npz`.

-  **Model:** A CNN with 3 convolutional layers and fully connected layers.

-  **Output:**

-  `saved_model.h5`: The trained model.

-  `label_classes.npy`: Encoded label classes.

  

### 3. Generating Guitar Tabs

The `generate_tabs.py` script predicts guitar notes and converts them into tablature:

-  **Input:** A file of a guitar note.

-  **Process:**

1. Extract the Mel-spectrogram of the input file.

2. Pass the spectrogram to the trained CNN for prediction.

3. Map the predicted note to its corresponding string and fret position using a predefined dictionary.

-  **Output:** The predicted note and tablature position.

  

---

  

## Dataset Information (https://www.kaggle.com/datasets/mohammedalkooheji/guitar-notes-dataset/data)

 **Notes:** Ranges from E2 (73.42 Hz) to G#5 (830.61 Hz).

 **Variations:**

-  **String Types:** Steel (`s`), Nylon (`n`).

-  **Plucking Styles:** Pick (`p`), Finger (`f`), Nail (`n`).

-  **Dynamics:** Normal (`n`), Loud (`l`), Muted (`m`).

 **Total Samples:** ~1500 recordings.

  

---

  

## How to Run

  

### 1. Preprocess the Dataset

```python preprocess.py```

  

Ensure the raw dataset is located at data/raw/Guitar Dataset. Feel free to add any

  

2. Train the Model


```python train_model.py```

This will save the model and labels in the models directory.

  

3. Generate Tabs

To predict a note and generate its guitar tab:

  

```python generate_tabs.py <audio_file.wav>```

  

Example Output:

```
Predicted Note: A#3 (B♭3)

Possible Tabs:

- String 5, Fret 13

- String 4, Fret 8

- String 3, Fret 3
```

  
  

# Requirements:

Python 3.8+

TensorFlow

NumPy

librosa

scikit-learn

Install dependencies with:

  

```pip install -r requirements.txt```

  

## Acknowledgements

This is a proof of concept so far! This currently has support for single note at a time, but we plan to add implementation for longer audio, detecing multiple notes over time, at tempo, and for multiple notes at a time for chords!
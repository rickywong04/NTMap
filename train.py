#!/usr/bin/env python3
"""
train_tabcnn_generator.py

PyTorch training with:
 - Progress bar per epoch using tqdm, including ms/step
 - Model summary using torchinfo.summary
 - Interactive Matplotlib plotting for Loss & Accuracy (Train / Val / Test)
 - Display each epoch's stats
 - Saves best model, final model
 - Dataset split 80/10/10
 - Memory-safe iteration (IterableDataset)
 - Train accuracy shown in progress bar
 - Test/Val accuracy after each epoch
 - Logging stats to CSV
 - Plots saved in "plots/training_plot_epoch_XXX.png"
 - LogSoftmax in model => use nn.NLLLoss
"""

import os
import csv
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

from tqdm import tqdm                    # For nice progress bars
from torchinfo import summary            # For model summary

# Local modules
from parse_jams import parse_guitarset_jams
from feature_extraction import extract_cqt_frames
from model_cnn import build_cnn

# ---------- Constants ----------
STRING_OPEN_MIDI = [40, 45, 50, 55, 59, 64]
MAX_FRET = 19
MUTED_IDX = 20

def fret_to_onehot(fret: int) -> np.ndarray:
    """Convert a fret in [-1..19] => one-hot of length 21."""
    arr = np.zeros(MAX_FRET+2, dtype=np.float32)  # 21
    if fret < 0:
        arr[MUTED_IDX] = 1
    else:
        arr[fret] = 1
    return arr

def split_files(wav_dir, test_ratio=0.1, val_ratio=0.1, seed=42):
    """Split .wav files 80/10/10 (train/val/test)."""
    all_files = [f for f in os.listdir(wav_dir) if f.endswith("_hex_cln.wav")]
    random.seed(seed)
    random.shuffle(all_files)
    total = len(all_files)
    test_count = int(total * test_ratio)
    val_count  = int(total * val_ratio)
    test_files = all_files[:test_count]
    val_files  = all_files[test_count:test_count+val_count]
    train_files= all_files[test_count+val_count:]
    return train_files, val_files, test_files

class GuitarSetIterable(IterableDataset):
    """
    Reads a list of .wav filenames. For each file, yields small mini-batches (X, Y).
    X => (B,1,192,9), Y => (B,6,21).
    """
    def __init__(self,
                 file_list,
                 wav_dir,
                 jams_dir,
                 batch_size=32,
                 sr=22050, hop_length=512,
                 n_bins=192, context_size=9):
        super().__init__()
        self.file_list   = file_list
        self.wav_dir     = wav_dir
        self.jams_dir    = jams_dir
        self.batch_size  = batch_size
        self.sr          = sr
        self.hop_length  = hop_length
        self.n_bins      = n_bins
        self.context_size= context_size

    def __iter__(self):
        for fname in self.file_list:
            wav_path = os.path.join(self.wav_dir, fname)
            base = fname.replace("_hex_cln.wav", "")
            jams_file = base + ".jams"
            jams_path = os.path.join(self.jams_dir, jams_file)
            if not os.path.exists(jams_path):
                # skip if .jams not found
                continue

            # Parse labels
            labels_6 = parse_guitarset_jams(jams_path, self.sr, self.hop_length)
            # Extract frames
            X_frames = extract_cqt_frames(
                wav_path, sr=self.sr, hop_length=self.hop_length,
                n_bins=self.n_bins, bins_per_octave=24, context_size=self.context_size
            )

            # Align length
            min_len = min(X_frames.shape[0], labels_6.shape[0])
            X_frames = X_frames[:min_len]
            labels_6 = labels_6[:min_len]

            # One-hot encode
            Y_all = []
            for i in range(min_len):
                row_oh = [fret_to_onehot(f) for f in labels_6[i]]
                Y_all.append(np.array(row_oh, dtype=np.float32))
            Y_all = np.array(Y_all, dtype=np.float32)

            # Now yield mini-batches
            idx = 0
            while idx < min_len:
                end = min(idx + self.batch_size, min_len)
                X_batch = X_frames[idx:end]
                Y_batch = Y_all[idx:end]
                idx = end

                # Convert to torch tensor
                X_batch_t = torch.from_numpy(X_batch)  # shape (B,1,192,9)
                Y_batch_t = torch.from_numpy(Y_batch)  # shape (B,6,21)
                yield (X_batch_t, Y_batch_t)

def get_steps_estimate(file_list, approx_frames=1000, batch_size=16):
    """Rough (#files * approx_frames) / batch_size => steps/epoch."""
    total_frames_est = len(file_list)*approx_frames
    return max(1, total_frames_est // batch_size)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure we have a directory for saving plots
    os.makedirs("plots", exist_ok=True)

    WAV_DIR = "data/raw/wav"
    JAMS_DIR= "data/raw/jams"

    # 1) Split the dataset
    train_files, val_files, test_files = split_files(
        WAV_DIR, test_ratio=0.1, val_ratio=0.1)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    if len(train_files) == 0:
        print("No training data. Exiting.")
        return

    # 2) Build model
    model = build_cnn(num_strings=6, num_frets=21)
    model.to(device)

    # 3) Model summary
    from torchinfo import summary
    summary(model, input_size=(1, 1, 192, 9))  # (batch=1, channels=1, 192 freq-bins, 9 context frames)

    # 4) Define Loss & Optimizer
    # Using NLLLoss, since we use nn.LogSoftmax in the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # 5) Create IterableDatasets & DataLoaders
    batch_size = 32
    train_ds = GuitarSetIterable(train_files, WAV_DIR, JAMS_DIR, batch_size=batch_size)
    val_ds   = GuitarSetIterable(val_files,   WAV_DIR, JAMS_DIR, batch_size=batch_size)
    test_ds  = GuitarSetIterable(test_files,  WAV_DIR, JAMS_DIR, batch_size=batch_size)

    train_loader = DataLoader(train_ds, batch_size=None)  # because we yield mini-batches ourselves
    val_loader   = DataLoader(val_ds,   batch_size=None)
    test_loader  = DataLoader(test_ds,  batch_size=None)

    # Estimate steps
    train_steps = get_steps_estimate(train_files, approx_frames=1000, batch_size=batch_size)
    val_steps   = get_steps_estimate(val_files,   approx_frames=1000, batch_size=batch_size)
    test_steps  = get_steps_estimate(test_files,  approx_frames=1000, batch_size=batch_size)
    print(f"Estimated steps => train:{train_steps}, val:{val_steps}, test:{test_steps}")

    # 6) Hyperparameters
    epochs = 512

    # 7) Setup CSV logging
    out_csv = open("training_stats.csv", "w", newline='')
    csv_writer = csv.writer(out_csv)
    csv_writer.writerow(["epoch","train_loss","val_loss","test_loss","train_acc","val_acc","test_acc"])

    # 8) Real-time interactive plots
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    best_val_acc = -1
    best_epoch   = -1

    # 9) Evaluate function
    def evaluate(loader, steps):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_count   = 0
        cstep = 0
        with torch.no_grad():
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)  # shape (B,6,21)

                # forward
                preds = model(X_batch)   # => (B,6,21) log-probs
                targ_class = Y_batch.argmax(dim=2).view(-1)  # => (B*6)
                logits = preds.view(-1, 21)                  # => (B*6,21)

                loss = criterion(logits, targ_class)
                total_loss += loss.item()

                pred_class = logits.argmax(dim=1)  # => (B*6)
                correct = (pred_class == targ_class).sum().item()
                total_correct += correct
                total_count   += targ_class.numel()

                cstep += 1
                if cstep >= steps:
                    break

        avg_loss = total_loss / cstep if cstep>0 else 0
        accuracy = total_correct / total_count if total_count>0 else 0
        return avg_loss, accuracy

    # 10) Main training loop
    for epoch in range(1, epochs+1):
        model.train()
        cstep = 0
        total_loss = 0
        total_correct = 0
        total_count   = 0

        step_time_start = time.perf_counter()

        # TQDM progress bar
        with tqdm(total=train_steps, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                preds = model(X_batch)  # => (B,6,21) (log-probs)
                targ_class = Y_batch.argmax(dim=2).view(-1)  # => (B*6)
                logits = preds.view(-1, 21)

                loss = criterion(logits, targ_class)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # partial accuracy
                pred_class = logits.argmax(dim=1)
                correct = (pred_class == targ_class).sum().item()
                total_correct += correct
                total_count   += targ_class.numel()

                cstep += 1

                # measure step time
                step_time_end = time.perf_counter()
                ms_per_step   = (step_time_end - step_time_start) * 1000.0
                step_time_start = time.perf_counter()

                train_acc_so_far = (total_correct / total_count) if total_count>0 else 0
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc":  f"{train_acc_so_far:.4f}",
                    "ms/step": f"{ms_per_step:.1f}"
                })

                pbar.update(1)
                if cstep >= train_steps:
                    break

        train_loss = total_loss / cstep if cstep>0 else 0
        train_acc  = (total_correct / total_count) if total_count>0 else 0

        # Evaluate val/test
        val_loss, val_acc   = evaluate(val_loader,   val_steps)
        test_loss, test_acc = evaluate(test_loader,  test_steps)

        # Store stats
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        # Write CSV
        csv_writer.writerow([epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc])
        out_csv.flush()

        # Check best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), "best_model.pt")

        # Print a summary line
        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

        # =============== Live Plot ===============
        axs[0].cla()
        axs[1].cla()

        epochs_arr = np.arange(1, epoch+1)

        # Loss subplot
        axs[0].plot(epochs_arr, train_losses, 'o-', label='Train Loss')
        axs[0].plot(epochs_arr, val_losses,   'o-', label='Val Loss')
        axs[0].plot(epochs_arr, test_losses,  'o-', label='Test Loss')
        axs[0].set_title('Loss vs. Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Accuracy subplot
        axs[1].plot(epochs_arr, train_accs, 'o-', label='Train Acc')
        axs[1].plot(epochs_arr, val_accs,   'o-', label='Val Acc')
        axs[1].plot(epochs_arr, test_accs,  'o-', label='Test Acc')
        axs[1].set_title('Accuracy vs. Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()
        plt.pause(0.01)  # allow interactive updates
        fig.savefig(f"plots/training_plot_epoch_{epoch:03d}.png")

    # End of training
    out_csv.close()
    print(f"\nFinished training. Best val_acc={best_val_acc:.4f} at epoch={best_epoch}.\n")

    # Evaluate final
    final_test_loss, final_test_acc = evaluate(test_loader, test_steps)
    print(f"Final test_acc={final_test_acc:.4f}, test_loss={final_test_loss:.4f}")

    # Save final
    torch.save(model.state_dict(), "final_model.pt")
    print("Saved final model as final_model.pt")

    plt.ioff()   # turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()

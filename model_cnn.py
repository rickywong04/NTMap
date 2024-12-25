#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class TabCNN(nn.Module):
    """
    A CNN that outputs shape (B, 6, 21) with log-probabilities 
    (via LogSoftmax) along the frets dimension for each of the 6 strings.
    """
    def __init__(self, num_strings=6, num_frets=21):
        super().__init__()
        self.num_strings = num_strings
        self.num_frets   = num_frets

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.dropout2 = nn.Dropout(0.25)

        # Fully connected layers
        # After two (2x2) pools for input (1,192,9), we get ~ (64,48,2) => flatten: 64*48*2=6144
        self.fc1 = nn.Linear(64 * 48 * 2, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_strings * num_frets)  # => (B,6*21)

        # We'll apply a log-softmax along the fret dimension
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        # x shape => (B,1,192,9)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = self.fc2(x)  # => (B, 6*21)
        x = x.view(-1, self.num_strings, self.num_frets)  # => (B,6,21)

        out = self.log_softmax(x)  # => (B,6,21) log-probs
        return out

def build_cnn(num_strings=6, num_frets=21):
    """Factory function to build the TabCNN model."""
    model = TabCNN(num_strings=num_strings, num_frets=num_frets)
    return model

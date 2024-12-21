#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn(
    input_shape=(192, 9, 1),
    num_strings=6,
    num_frets=20  # 0..19 + 1 for muted => if we treat muted as index=20
):
    """
    Return a Keras model that produces shape (batch_size, 6, num_frets)
    with a softmax over each string's fret distribution.
    """
    model = models.Sequential()

    # Convolution stack
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Flatten + Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output dimension: 6 * num_frets
    output_dim = num_strings * num_frets
    model.add(layers.Dense(output_dim, activation='linear'))
    model.add(layers.Reshape((num_strings, num_frets)))
    model.add(layers.Softmax(axis=-1))

    # Compile
    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

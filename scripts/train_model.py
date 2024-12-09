#!/usr/bin/env python3
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, callbacks, utils

DATA_NPZ = "data/processed/processed_data.npz"
MODEL_PATH = "models/saved_model.h5"
LABELS_PATH = "models/label_classes.npy"

def load_data():
    data = np.load(DATA_NPZ, allow_pickle=True)
    X = data['X']  # shape: (num_samples, 128, time_frames, 1)
    y = data['y']
    classes = data['classes']
    return X, y, classes

def build_cnn(input_shape, num_classes):
    model = models.Sequential()
    # A simple CNN architecture
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X, y, classes = load_data()
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(classes)
    y_cat = utils.to_categorical(y_enc, num_classes=num_classes)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = build_cnn(X_train.shape[1:], num_classes)
    model.summary()

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[es]
    )

    # Save model and labels
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    np.save(LABELS_PATH, le.classes_)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()

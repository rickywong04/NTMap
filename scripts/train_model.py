#!/usr/bin/env python3
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers, callbacks, utils, optimizers

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

    # -------------------------------------------------
    # 1) Example: Add or change Conv layers
    # -------------------------------------------------
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Optionally, you can try more layers or different filter sizes:
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Optionally add another convolutional layer
    # model.add(layers.Conv2D(256, (3,3), activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    # -------------------------------------------------
    # 2) Example: Modify Dense layers
    # -------------------------------------------------
    model.add(layers.Dense(256, activation='relu'))  # Increased from 128 to 256
    model.add(layers.Dropout(0.3))

    # Optionally add another Dense layer
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))

    # -------------------------------------------------
    # 3) Example: Change the optimizer or learning rate
    # -------------------------------------------------
    # Option A: Keep Adam but lower the learning rate:
    # opt = optimizers.Adam(learning_rate=1e-4)

    # Option B: Use RMSProp with a custom learning rate:
    # opt = optimizers.RMSprop(learning_rate=1e-4)

    # Option C: Use SGD with momentum:
    # opt = optimizers.SGD(learning_rate=1e-2, momentum=0.9)

    # If you do not define one of these, you’ll just use the default from below:
    opt = optimizers.Adam(learning_rate=1e-3)

    # -------------------------------------------------
    # 4) Example: Change the loss function
    # -------------------------------------------------
    # Common for multi-class classification:
    # "categorical_crossentropy" is typical if y is one-hot encoded
    # "sparse_categorical_crossentropy" if y is not one-hot encoded
    # If you want to try a different approach:
    # loss_fn = 'categorical_crossentropy'
    # loss_fn = 'kl_divergence'   # Not typical, but possible
    # loss_fn = 'focal_loss'      # Requires additional libraries
    loss_fn = 'categorical_crossentropy'

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    return model

def main():
    # 1) Load data
    X, y, classes = load_data()

    # 2) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(classes)
    y_cat = utils.to_categorical(y_enc, num_classes=num_classes)

    # 3) Split off a test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_cat, 
        test_size=0.1, 
    )

    # 4) Split train_val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=0.1111,  # ~0.1 / (1 - 0.1) => 0.1111
    )
    # So now: ~80% train, 10% val, 10% test

    # 5) Build and summarize your model
    model = build_cnn(X_train.shape[1:], num_classes)
    model.summary()

    # 6) Early stopping
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 7) Train model using train and val sets
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=16,
        callbacks=[es]
    )

    # 8) Evaluate on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # 9) Evaluate on the test set (held out)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # 10) Save model and labels
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    np.save(LABELS_PATH, le.classes_)

if __name__ == "__main__":
    main()

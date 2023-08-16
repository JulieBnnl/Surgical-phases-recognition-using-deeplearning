import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
import tensorflow as tf
import imageio
from tensorflow.keras.models import load_model

def run_experiment(label_processor,train_data,train_masks,train_labels,test_data,test_masks,test_labels):
    filepath = "tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    EPOCHS = 30
    seq_model = get_sequence_model(label_processor)
    history = seq_model.fit(
        [train_data, train_masks],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data, test_masks], test_labels)
    acc = round(accuracy * 100, 2)
    print(f"Test accuracy: {acc}%")

    return history, seq_model, acc

def get_sequence_model(label_processor):
    
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


def classifier():
    print("Model preparation...")
    IMG_SIZE = 1000
    BATCH_SIZE = 64
    EPOCHS = 30


    train_df = pd.read_csv("csv/train.csv")
    test_df = pd.read_csv("csv/test.csv")

    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
    print(label_processor.get_vocabulary())

    # Load NPY files for training data
    train_data = np.load("npy/train_frame_features.npy")
    train_masks = np.load("npy/train_frame_masks.npy")
    train_labels = np.load("npy/train_labels.npy")

    # Load NPY files for test data
    test_data = np.load("npy/test_frame_features.npy")
    test_masks = np.load("npy/test_frame_masks.npy")
    test_labels = np.load("npy/test_labels.npy")

    image_height = 224
    image_width = 224
    num_channels = 3
    num_classes = 3
    acc = 0
    best_acc= 0

    for i in range(20):
        _, sequence_model,acc = run_experiment(label_processor,train_data,train_masks,train_labels,test_data,test_masks,test_labels)
        if acc>best_acc:
            sequence_model.save("model")
            best_acc = acc

    print(f"The best model for accuracy {best_acc} has been recorded!")
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
import csv

def classification(model):
    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 100

    train_path = "train"
    dataset_path = os.listdir(train_path)
    test_path = "test"

    rooms = []

    for item in dataset_path:
        # Get all the file names
        all_rooms = os.listdir(train_path + '/' +item)

        # Add them to the list
        for room in all_rooms:
            rooms.append((item, str(train_path + '/' +item) + '/' + room))

    train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])

    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))

    feature_extractor = build_feature_extractor()

    # Path to directory containing video sequences
    sequences_dir = "dataset_to_be_labeled"
    sequence_model=load_model(model)

    # List to store label predictions
    predictions = []

    fichier_csv = "csv/results_predict.csv"

    # Browse video sequence files in the directory in ascending order
    for filename in sorted(os.listdir(sequences_dir), key=lambda x: int(x.split("_")[1].split(".")[0])):
       
        sequence_path = os.path.join(sequences_dir, filename)
        predic = sequence_prediction(sequence_path,label_processor,test_path,sequence_model,feature_extractor)
    
    # Write the prediction to the result CSV file.
        with open(fichier_csv, mode='a', newline='') as fichier:
            writer = csv.writer(fichier)
            writer.writerow([filename, predic])

        print(filename + " : " + predic)

    # Create a pandas DataFrame for results
    sequence_numbers = range(len(predictions))
    df = pd.DataFrame({"Sequence": sequence_numbers, "Label": predictions})


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames,feature_extractor):
    
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path,label_processor,test_path,sequence_model,feature_extractor):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join(test_path, path))
    frame_features, frame_mask = prepare_single_video(frames,feature_extractor)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    max_index = 0

    for j in range(1, len(probabilities)):
        if probabilities[j] > probabilities[max_index]:
            max_index = j
    print(class_vocab[max_index])

    return class_vocab[max_index]

def build_feature_extractor():
    IMG_SIZE=224
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")



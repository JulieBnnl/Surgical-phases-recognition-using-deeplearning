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
from tqdm import tqdm
from tensorflow.keras.models import load_model


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

def prepare_all_videos(df, root_dir, save_prefix,label_processor,feature_extractor):
    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()

    # Take all classlabels from train_df column named 'tag' and store in labels
    labels = df["tag"].values

    # Convert classlabels to label encoding
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool") # 145,20
    frame_features = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32") #145,20,2048

    # For each video.
    for idx, path in enumerate(tqdm(video_paths,desc="Processing videos", unit="video")):
        
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    np.save(save_prefix + 'frame_features.npy', frame_features)
    np.save(save_prefix + 'frame_masks.npy', frame_masks)
    np.save(save_prefix + 'labels.npy', labels)

    return (frame_features, frame_masks), labels

def prepare_videos():
    print("Videos preparation...")
    train_path = "train"
    test_path = "test"
    os.makedirs("npy",exist_ok=True)
    os.makedirs("csv", exist_ok=True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        except RuntimeError as e:
            print(e)


    IMG_SIZE=224

    dataset_path = os.listdir('train')

    label_types = os.listdir('train')
    print (label_types)

    rooms = []

    for item in dataset_path:
        # Get all the file names
        all_rooms = os.listdir(train_path + '/' +item)

        # Add them to the list
        for room in all_rooms:
            rooms.append((item, str(train_path + '/' +item) + '/' + room))

    # Build a dataframe
    train_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
    print(train_df.head())
    print(train_df.tail())

    df = train_df.loc[:,['video_name','tag']]
    df
    df.to_csv('csv/train.csv')

    dataset_path = os.listdir(test_path)
    print(dataset_path)

    room_types = os.listdir(test_path)
    print("Types of activities found: ", len(dataset_path))

    rooms = []

    for item in dataset_path:
        # Get all the file names
        all_rooms = os.listdir(test_path + '/' +item)

        # Add them to the list
        for room in all_rooms:
            rooms.append((item, str(test_path + '/' +item) + '/' + room))

    # Build a dataframe
    test_df = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])
    print(test_df.head())
    print(test_df.tail())

    df = test_df.loc[:,['video_name','tag']]
    df
    df.to_csv('csv/test.csv')

    train_df = pd.read_csv("csv/train.csv")
    test_df = pd.read_csv("csv/test.csv")

    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for testing: {len(test_df)}")


    train_df.sample(10)

    IMG_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 100

    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048

    feature_extractor = build_feature_extractor()

    label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
    print(label_processor.get_vocabulary())

    labels = train_df["tag"].values
    labels = label_processor(labels[..., None]).numpy()


    train_data, train_labels = prepare_all_videos(train_df, "train","npy/train_",label_processor,feature_extractor)
    test_data, test_labels = prepare_all_videos(test_df, "test","npy/test_",label_processor,feature_extractor)

    print(f"Frame features in train set: {train_data[0].shape}")
    print(f"Frame masks in train set: {train_data[1].shape}")
    print("Videos ready !")
import os
import random
import shutil
from tqdm import tqdm


def train_test():
    print("Moving videos in progress ...") 

    # Train and test folder path
    train_path = "train"
    test_path = "test"
    dataset_path = "dataset"

    # Creation of train and test folders
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # List of classes in dataset folder
    subdirectories = [subdir for subdir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subdir))]

    for subdir in subdirectories:
        # Path to class subfolder in dataset
        subdir_path = os.path.join(dataset_path, subdir)

        # Path to class subfolder in train and test
        train_subdir_path = os.path.join(train_path, subdir)
        test_subdir_path = os.path.join(test_path, subdir)

        # Create class subfolder in train and test
        os.makedirs(train_subdir_path, exist_ok=True)
        os.makedirs(test_subdir_path, exist_ok=True)

        # List of videos in the class subfolder
        videos = os.listdir(subdir_path)

        # Random shuffle of the video list
        random.shuffle(videos)

        # Calculation of number of training videos (70%)
        train_size = int(0.7 * len(videos))

        # Moving videos into train and test subfolders
        for i, video in enumerate(tqdm(videos, desc=f"Processing {subdir}", unit="video")):
            
            video_path = os.path.join(subdir_path, video)
            if i < train_size:
                # Move to train subfolder
                destination = os.path.join(train_subdir_path, video)
            else:
                # Move to test subfolder
                destination = os.path.join(test_subdir_path, video)
            shutil.copy2(video_path, destination)

    print("Moving videos completed !") 

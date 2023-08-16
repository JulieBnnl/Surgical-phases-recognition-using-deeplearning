import os
import random
from collections import defaultdict

def egalize(dataset_folder):
    print("Equalizing the number of videos...")
     

    ## Browse the subfolders of the "dataset" folder and count the number of videos in each subfolder
    video_counts = defaultdict(int)
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".mp4"):
                subfolder = os.path.basename(root)
                video_counts[subfolder] += 1

    ## Find the minimum number of videos among all subfolders
    min_video_count = min(video_counts.values())


    # Randomly delete videos in each subfolder so that they have the same number of videos
    for subfolder, count in video_counts.items():
        if count > min_video_count:
            videos = os.listdir(os.path.join(dataset_folder, subfolder))
            videos_to_delete = random.sample(videos, count - min_video_count)
            for video in videos_to_delete:
                video_path = os.path.join(dataset_folder, subfolder, video)
                os.remove(video_path)

    print("Equalization complete .")
    print(f"Number of videos per class :{min_video_count}")


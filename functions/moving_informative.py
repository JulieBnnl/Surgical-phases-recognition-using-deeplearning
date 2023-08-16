import os
import shutil
import pandas as pd

def copy_informative_videos(csv_file_path, source_videos_folder, destination_folder):
    # Load CSV file using pandas without header
    df = pd.read_csv(csv_file_path, header=None)

    # Filter rows where the class is "Informative"
    informative_videos = df[df.iloc[:, 1] == 'Informative']

    # Iterate through each row and copy videos to the destination folder
    for index, row in informative_videos.iterrows():
        video_name = row.iloc[0]  # First column is video name
        video_source_path = os.path.join(source_videos_folder, video_name)
        video_destination_path = os.path.join(destination_folder, video_name)
        
        # Check if the video exists and copy it to the destination folder
        if os.path.exists(video_source_path):
            shutil.copy(video_source_path, video_destination_path)
            print(f"Copied: {video_name}")
        else:
            print(f"Video not found: {video_name}")

    print("Copying informative videos completed!")

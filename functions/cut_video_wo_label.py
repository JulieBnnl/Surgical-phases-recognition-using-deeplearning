import cv2
import os
from tqdm import tqdm

# Function to cut video into 30-frame sequences
def split_video(video_path, output_dir):
    print("Cutting the video...")
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    sequence_count = 1
    sequence_images = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        sequence_images.append(frame)
        count += 1
        if count % 30 == 0:
            # Save the sequence of 30 images as a video
            sequence_output_path = os.path.join(output_dir, f"sequence_{sequence_count}.mp4")
            height, width, _ = sequence_images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(sequence_output_path, fourcc, 30.0, (width, height))
            for image in sequence_images:
                out.write(image)
            out.release()
            sequence_count += 1
            sequence_images = []
    cap.release()
    print("Successfully cut video !")

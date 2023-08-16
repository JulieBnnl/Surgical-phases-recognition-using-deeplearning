import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

def create_video_sequences(video_path, xml_path, output_folder):
    print("Cutting the video into sequences...")
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the XML file to obtain the labels for each frame
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_labels = {}  # Dictionary to store labels for each frame
    for image in root.findall('image'):
        frame_num = int(image.get('id'))
        tag_element = image.find('tag')

        if tag_element is not None:
            label = tag_element.get('label')
            frame_labels[frame_num] = label

    # Load main video
    video = cv2.VideoCapture(video_path)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize variables
    frame_counter = 0
    current_sequence = []
    current_label = None
    video_counter = 1
    for i in tqdm(range(frame_count)):
        # Read the frame
        success, frame = video.read()
        if not success:
            break

        # Get the label of the current frame
        frame_counter += 1
        label = frame_labels.get(frame_counter)

        # Check if the frame is labelled
        if label is not None:
        # Check if the label of the current frame is different from the current sequence
            if label != current_label or len(current_sequence) == 30: # Maximum sequence size: 30 frames
            # Create a video from the frames of the current sequence
                if current_sequence:
                    create_video_from_frames(current_sequence, current_label, output_folder, video_counter)
                    video_counter += 1

                # Reset current sequence and label
                current_sequence = []
                current_label = label

            # Add frame to current sequence
            current_sequence.append(frame)

    # Create a video from the frames of the remaining sequence
    if current_sequence:
        create_video_from_frames(current_sequence, current_label, output_folder, video_counter)

    # Freeing up resources
    video.release()
    print("Video successfully cut !")


def create_video_from_frames(frames, label, output_folder, video_counter):
    # Create output folder for label if none exists
    label_output_folder = os.path.join(output_folder, label)
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder)

    # Counter-based video name
    video_name = f"{label}_{video_counter}"

    # Video output path
    output_path = os.path.join(label_output_folder, f"{video_name}.mp4")

    # Get frame size from first frame
    height, width, _ = frames[0].shape

    # Create the VideoWriter object to write the video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

    # Writing frames in video
    for frame in frames:
        out.write(frame)

    # Freeing up resources
    out.release()


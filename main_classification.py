"""
Author: BONNAIL Julie
Date: 2023/08/04
Lab/Organization: NearLab - Politecnico di Milano
"""

from functions import cut_video_wo_label, classification, moving_informative

# Path to video to be labeled
video_path = "videos/160.mp4"
# Output directory for sequences
output_dir = "dataset_to_be_labeled"

# Cut the video to be labeled into 30-frame sequences
cut_video_wo_label.split_video(video_path,output_dir)
# The result is stored in dataset_to_be_labeled

# Classification of video sequences according to the model previously trained by main_classifier.py
# The chosen model can also be pretrained_model or pretrained_non_info_model.
model = "model/"
classification.classification(model)
# Results are saved in csv/results_predict.csv

# Moving sequences classified as "Informative" to the informative folder
csv_path = "csv/results_predict.csv"
moving_informative.copy_informative_videos(csv_path, output_dir+'/', "informative")


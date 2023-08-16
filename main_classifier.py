"""
Author: BONNAIL Julie
Date: 2023/08/04
Lab/Organization: NearLab - Politecnico di Milano
"""

from functions import imports, cut_video, train_test, delete_doss, egalize, prepare_videos, classifier


# Installation of the required package tensorflow.docs
imports.install_docs()


# Splitting the hand_annotated video into smaller video sequences of the same label, each containing a maximum of 30 frames."
labeled_video_path = "videos/160.mp4"
xml_path = "xml/160.xml"
output_folder = "dataset"
cut_video.create_video_sequences(labeled_video_path, xml_path, output_folder)
# The result is stored in dataset

# Equalize the number of videos in each folder so that the model learns correctly
egalize.egalize("dataset")

# Divide dataset into training and test data (70% / 30%)
train_test.train_test()
#Results are stored in the train and test folders respectively. 

# Deleting the dataset folder, which is now useless
delete_doss.delete_doss("dataset")

# Preparing videos for model training
prepare_videos.prepare_videos()

# Training the rnn model
classifier.classifier()
# The final model is saved in model
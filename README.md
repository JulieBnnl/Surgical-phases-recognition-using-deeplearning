# Surgical phases recognition using deeplearning
_(Julie Bonnail)_

Classification algorithm for informative videoscoloscopic by RNN

## Get started 

This project contains three programs. The first (main_classifier.py) allows you to create your own colonoscopy video classifier. The second (main_classification.py) classifies a colonoscopy video using a classifier. Finally, the third (main_comparison.py) is used to evaluate the classification performed.

## Architecture

This code is made up of several folders:
- csv: contains all the csv files created by the various programs
- dataset: contains videos sorted by label for training an RNN model
- dataset_to_be_labeled: contains videos to be labeled divided into 30-frame sequences
- functions : contains all the python functions required for the various programs
- informative: contains videos classified as "informative" by a model
- model: contains the model trained by main_classifer.py
- npy : contains numpy values useful for training the model
- pretrained_model and pretrained_non_info_model: contain RNN models already trained for colonoscopic video classification.
- test and train: contain the training and test data required to train the model
- videos: contains every video required for the various programs
- xml: contains video labeling files

### main_classifier.py

This program requires a previously labeled video (e.g. on CVAT) in an xml file. 
It splits the video into sequences of a maximum of 30 frames of the same label, which it saves in the "dataset" folder. It equalizes the number of videos in each class to avoid underfitting. It then divides this folder into training and test data (70%-30%). 
Finally, the program prepares each video and trains its RNN video classification model. This model is saved in the "model" folder.

### main_classification.py

This program needs a video to classify and an RNN model to run. It cuts the video into 30-frame sequences and saves them in "dataset_to_be_labeled".
Then, it classifies each sequence using a model. This model can be the one created with main_classifier.py, but also "pretrained_model" or "pretrained_non_info_model". "pretrained_model" has been trained with six classes: "Endoscope out of body", "Water flush", "Stuck against a wall", "Out of focus", "Tool" and "Informative". This model has an accuracy of 62.9%. "pretrained_non_info_model" was trained only on 3 classes: "Endoscope outof body", "Informative", and "Non informative". This model has an accuracy of 81.5%.
The classification result is a "results_predict" csv file saved in "csv". Videos classified as "Informative" are stored in the "informative" folder.

## main_comparison.py

This program needs a csv file corresponding to the result of a classification, an xml file of the manual labeling of the same video and a list of all the classes present. 
It creates a "temoin.csv" csv file, for which in each sequence of 30 frames, the most present label in the csv file has defined the sequence class.
It returns a csv file "comparison.csv" including in each box "0" if the same sequence received the same classification in two different ways. It writes the percentage of correct classifier prediction and displays the comparison matrices.


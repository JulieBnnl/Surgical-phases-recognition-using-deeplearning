"""
Author: BONNAIL Julie
Date: 2023/08/04
Lab/Organization: NearLab - Politecnico di Milano
"""

from functions import comparison

# Model comparison with hand labeling
xml_file_path = "xml/160.xml"
csv_to_test = "csv/results_predict.csv"
classes = ["Informative","Non informative", "Endoscope out of body"]

comparison.comparison_csv(xml_file_path,csv_to_test,classes)
# This creates a comparison csv file between the manually annotated video and the automatically classified one.
# The normal csv file includes a 0 if the classes are identical, otherwise it indicates the correct class and then the one detected.
# This also creates the confusion matrices


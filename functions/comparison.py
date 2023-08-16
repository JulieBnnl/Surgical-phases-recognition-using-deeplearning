import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import xml.etree.ElementTree as ET

 
def extract_labels(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    frame_sequences = []
    current_sequence = []
    current_label_counts = {}

    for image_elem in root.findall("image"):
        tag_elem = image_elem.find("tag")
        if tag_elem is not None:
            label = tag_elem.get("label")
            frame_id = int(image_elem.get("id"))
            current_sequence.append(label)

            if frame_id % 30 == 29:  # Check if 30 frames have been processed
                current_label_counts = {label: current_sequence.count(label) for label in current_sequence}
                most_common_label = max(current_label_counts, key=current_label_counts.get)
                frame_sequences.append((frame_id // 30, most_common_label))
                current_sequence = []

    return frame_sequences

def write_to_csv(frame_sequences, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        for sequence_num, most_common_label in frame_sequences:
            seq_num = str(sequence_num+1)
            writer.writerow(["sequence_"+seq_num+".mp4", most_common_label])

def comparison_csv(xml_file_path,test_file, classes):
    extracted_sequences = extract_labels(xml_file_path)
    write_to_csv(extracted_sequences, "csv/temoin.csv")
    compare_csvfiles("csv/temoin.csv", test_file, "csv/comparison.csv",classes)


def compare_csvfiles(temoin_file, test_file, result_file,classes):
    tot = 0
    err = 0
    true = []
    predict = []
    with open(temoin_file, newline='') as f1, open(test_file, newline='') as f2, open(result_file, 'w', newline='') as result_f:
        reader_temoin = csv.reader(f1)
        reader_test = csv.reader(f2)
        writer = csv.writer(result_f)

        for row_temoin, row_test in zip(reader_temoin, reader_test):
            compared_row = []
            for cell_temoin, cell_test in zip(row_temoin, row_test):
                tot += 1
                true.append(cell_temoin)
                predict.append(cell_test)

                if cell_temoin != cell_test:
                    err += 1
                    compared_row.append(f"{cell_temoin}>{cell_test}")
                else:
                    compared_row.append("0")
            writer.writerow(compared_row)
    pourcent_err = (err * 100) / tot
    print("Percentage of accuracy : " + str(pourcent_err) + "%")
    
    plot_confusion_matrices(true, predict,classes)


def plot_confusion_matrices(y_true, y_pred,classes, labels=["0", "1"]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    plt.suptitle("Confusion Matrices", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < len(classes):
            cls = classes[i]
            bin_true = [1 if yt == cls else 0 for yt in y_true]
            bin_pred = [1 if yp == cls else 0 for yp in y_pred]
            cm = confusion_matrix(bin_true, bin_pred)
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            im = ax.imshow(cm_percentage, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            ax.set(xticks=np.arange(len(labels)),
                   yticks=np.arange(len(labels)),
                   xticklabels=labels, yticklabels=labels,
                   xlabel="Predicted",
                   ylabel="True",
                   title="Confusion matrix for " + cls)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, "{:.2f}%".format(cm_percentage[i, j] * 100), ha="center", va="center", color="black")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




from matplotlib import pyplot as plt
import numpy as np

import csv

def load_csv_data(filepath):
    data = {}
    with open(filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None) # skip header
        
        # Load csv data into dict sorted by label
        for row in csv_reader:
            label = row[0]
            text = row[1]
            if label in data.keys():
                data[label].append(text)
            else:
                data[label] = [text]
        
    return data


def plot_label_distribution():
    # Load data
    data = load_csv_data("data/dialog_acts.csv")
    no_dupe_data = load_csv_data("data/no_duplicates_dialog_acts.csv")
    
    # Get label counts
    label_counts = {}
    for label, sentences in data.items():
        label_counts[label] = len(sentences)

    # Get label counts without duplicates
    no_dupe_label_counts = {}
    for label, sentences in no_dupe_data.items():
        no_dupe_label_counts[label] = len(sentences)
        
    # Plot data
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Label Distribution')
    axs[0].bar(label_counts.keys(), label_counts.values())
    axs[0].set_title("With Duplicates")
    axs[1].bar(no_dupe_label_counts.keys(), no_dupe_label_counts.values())
    axs[1].set_title("Without Duplicates")
    plt.show()
    
    return

if __name__ == '__main__':
    plot_label_distribution()
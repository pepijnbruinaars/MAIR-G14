import csv
import torch 
import pickle
import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
cwd = os.getcwd()
sys.path.append(cwd + "/src")

from helpers import load_csv_data
from intent_models.baselines.keyword_matching import match_sentence
from intent_models.ml_models.mlp import FeedForwardNN
from intent_models.ml_models.mlp import vectorize_data
from intent_models.ml_models.mlp import calculate_accuracy
from intent_models.ml_models.mlp import get_string_labels
from intent_models.ml_models.mlp import get_labels



def test_models():
    print("Running test suite...")
    # Load data
    test_data_dupes = load_csv_data("data/splits/test_dialog_acts.csv")
    test_data_no_dupes = load_csv_data("data/splits/test_dialog_acts_no_dupes.csv")

    # TODO: Test majority model

    # Test keyword model
    for data in [test_data_dupes, test_data_no_dupes]:
        correct_count = 0
        total_count = 0
        incorrect = []
        predictions = []
        for label, sentences in data.items():
            for sentence in sentences:
                prediction = match_sentence(sentence)
                predictions.append(prediction)
                total_count += 1
                # Anything which does not match a keyword should be represented like in the csv
                if prediction == 0:
                    prediction = "null"
                # Check if prediction matches label
                if prediction == label:
                    # print(f"Keyword model predicted {prediction} for \"{sentence}\"
                    # when it should have predicted {label}")
                    correct_count += 1
                else:
                    incorrect.append((sentence, label, prediction))

        # Save incorrect predictions to csv
        with open("data/results/incorrect_predictions.csv", "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            csv_writer.writerow(["sentence", "label", "prediction"])
            for incorrect_prediction in incorrect:
                csv_writer.writerow(incorrect_prediction)
        print(f"Keyword model accuracy: {correct_count/total_count}")
        prediction_count = {}
        for prediction in predictions:
            if prediction in prediction_count:
                prediction_count[prediction] += 1
            else:
                prediction_count[prediction] = 1
        print(f"Prediction counts: {prediction_count}")

    # TODO: Test tree model
    
    # TODO: Test neural model

    with open("models/vectorizer.pkl","rb") as file:
        vectorizer = pickle.load(file)

    test_data = pd.read_csv("data/splits/test_dialog_acts_no_dupes.csv")
    
    x, y = vectorize_data(test_data)

    VECTOR_SIZE = x.shape[1]
    OUTPUT_SIZE = 15  # 15 different dialog acts
    HIDDEN_SIZE = 75
    DROPOUT_RATE = 0.2

    xix = pd.read_csv("data/dialog_acts.csv")
    counted = xix.groupby('label').size().reset_index(name='count')

    print(counted)
    #Load the no duplicates model
    def test_mlp_model(duplicates = False):

        model = FeedForwardNN(VECTOR_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
        if duplicates == False:
            model.load_state_dict(torch.load("models/mlp_model.pt"))
            test_data = pd.read_csv("data/splits/test_dialog_acts_no_dupes.csv")

        else:
            model.load_state_dict(torch.load("models/mlp_model_dupes.pt"))
            test_data = pd.read_csv("data/splits/test_dialog_acts.csv")
        model.eval()

        x, y = vectorize_data(test_data)
        logits = model(x)
        preds = [torch.argmax(z).item() for z in logits]
        print(calculate_accuracy(preds,y.tolist()))

        preds = get_string_labels(preds)
        y = get_string_labels(y.tolist())

        disp = ConfusionMatrixDisplay.from_predictions(y, preds, 
                labels= get_labels(),  xticks_rotation= "vertical")
        
    

        precision, recall, fscore, support = precision_recall_fscore_support(preds,y, labels=get_labels())
        
        def occurences_count_plot():
            counter = Counter(y)
            counts = [counter[label] for label in get_labels()]
            print(counts)
            plt.figure(figsize=(10, 6))
            plt.bar(get_labels(), counts, color='green')

        
            # Adding labels and title
            plt.xlabel('Classes')
            plt.ylabel('Number of Occurences in Test Set')
            plt.title('Count of Occurences per Class in Test Set')

            # Rotating x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()


            plt.show() 
        occurences_count_plot()
    
    test_mlp_model(duplicates=False)


test_models()
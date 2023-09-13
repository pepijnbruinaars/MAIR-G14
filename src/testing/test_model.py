import csv
from helpers import load_csv_data
from keyword_matching.keyword_matching import match_sentence


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
        for label, sentences in data.items():
            for sentence in sentences:
                prediction = match_sentence(sentence)
                total_count += 1
                # Anything which does not match a keyword should be represented like in the csv
                if prediction == 0:
                    prediction == "null"
                # Check if prediction matches label
                if prediction == label:
                    # print(f"Keyword model predicted {prediction} for \"{sentence}\" when it should have predicted {label}")
                    correct_count += 1
                else:
                    incorrect.append((sentence, label, prediction))
                    
        # Save incorrect predictions to csv
        with open("data/results/incorrect_predictions.csv", 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(["sentence", "label", "prediction"])
            for incorrect_prediction in incorrect:
                csv_writer.writerow(incorrect_prediction)
        print(f"Keyword model accuracy: {correct_count/total_count}")
    
    # TODO: Test tree model
    # TODO: Test neural model
    return
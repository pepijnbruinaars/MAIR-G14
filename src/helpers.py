import argparse
import os
from nltk.corpus import stopwords
import string
import nltk
import csv

from intent_models.ml_models.random_forest import generate_random_forest


def load_csv_data(filepath):
    data = {}
    with open(filepath, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip header

        # Load csv data into dict sorted by label
        for row in csv_reader:
            label = row[0]
            text = row[1]
            if label in data.keys():
                data[label].append(text)
            else:
                data[label] = [text]

        return data


def save_data_as_csv(filepath, data_dict):
    with open(filepath, "w") as f:
        write = csv.writer(f)

        write.writerow(["label", "text"])
        for key in data_dict.keys():
            for entry in data_dict[key]:
                row = [key]
                row.append(entry)
                write.writerow(row)


def remove_stopwords(data_dict):
    stop_words = set(stopwords.words("english"))

    for key, entries in data_dict.items():
        data_dict[key] = [
            " ".join([word for word in entry.split() if word not in stop_words])
            for entry in entries
        ]

    return data_dict


def prep_user_input(user_input: str):
    # Remove stopwords
    try:
        user_input = " ".join(
            [word for word in user_input.split() if word not in stopwords.words("english")]
        )
    except LookupError:
        print("Stopwords have not yet been downloaded. Downloading now...")
        nltk.download('stopwords')
        user_input = " ".join(
            [word for word in user_input.split() if word not in stopwords.words("english")]
        )

    # Remove punctuation
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))

    # Convert to lowercase
    user_input = user_input.lower()

    return user_input


def check_models(args: argparse.Namespace):
    """Check if the models folder contains the necessary models for the selected model.
    If not, we train the selected model.

    Args:
        args (argparse.Namespace): The arguments passed to the program.

    Raises:
        NotImplementedError: Raised if the selected model is not implemented yet.
        ValueError: Raised if the selected model is invalid.
    """
    # Check models folder for first time use
    try:
        os.listdir("models")
    except FileNotFoundError:
        os.mkdir("models")

    # Check for each model
    if args.intent_model == "RF":
        with os.scandir("models") as folder:
            # If folder contains optimized_random_forest.joblib, then we are good to go
            if "optimized_random_forest.joblib" in [file.name for file in folder]:
                pass
            # Train model if not
            else:
                generate_random_forest()
        return

    if args.intent_model == "neural":
        raise NotImplementedError(
            "Neural model not implemented yet. Please select another model."
        )

    if args.intent_model == "majority" or args.model == "keyword":
        return

    # If we get here, the model does not exist
    raise ValueError(f"Invalid model: {args.model}")

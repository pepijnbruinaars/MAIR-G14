from intent_models.ml_models.random_forest import generate_random_forest
from intent_models.ml_models.mlp import fit_mlp
from nltk.corpus import stopwords
import argparse
import string
import nltk
import csv
import os
import re
import pandas as pd
import random


def get_suggestion_prefixes():
    return [
        "I suggest you go to",
        "I recommend",
        "I would recommend",
        "I would suggest",
        "You could go to",
        "You could try",
        "You could visit",
        "You might like",
        "You might enjoy",
        "You might want to try",
        "You might want to visit",
        "You might want to go to",
        "You might want to check out",
        "You might want to go to",
    ]


def get_message_templates():
    return {
        # Intent messages
        "welcome": (
            """Hello, I am here to help you find a suitable restaurant \N{rocket}.
            \r\tTo exit, just type 'exit'!
            \r\tI can find you a restaurant using the following preferences:
            - food \N{fork and knife}
            - area \N{house building}
            - price range \N{money bag}"""
        ),
        "hello": "\N{waving hand sign} Hi! How can I help you?",
        "thankyou": "You're welcome! \N{grinning face with smiling eyes}",
        # Error messages
        "err_req": (
            "I'm sorry \N{pensive face}, I can't answer your question because I don't know"
            " any restaurants that match your preferences."
        ),
        "err_inf_no_result": "I'm sorry \N{pensive face}, I can't find any restaurants that match your preferences.",
        "err_neg_next_step": "I'm sorry \N{pensive face}, I can't help you with that.",
        "err_neg_no_options": "I'm sorry \N{pensive face}, there are no other options that match your preferences.",
    }


def get_identity(gender):
    emoji = "\N{robot face}"
    name = "Bot"
    match gender:
        case "male" | "man":
            emoji = "\N{man}"
            name = "John"
        case "female" | "woman":
            emoji = "\N{woman}"
            name = "Jane"
        case _:  # Default (null) case
            emoji = "\N{robot face}"
            name = "Bot"
    return emoji, name


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
            [
                word
                for word in user_input.split()
                if (
                    word not in stopwords.words("english")
                    or word == "no"
                    or word == "yes"
                    or word == "more"
                    or word == "not"
                )
            ]
        )
    except LookupError:
        print("Stopwords have not yet been downloaded. Downloading now...")
        nltk.download("stopwords")
        prep_user_input(user_input)

    # Remove punctuation
    user_input = user_input.translate(str.maketrans("", "", string.punctuation))

    # Convert to lowercase
    user_input = user_input.lower()

    return user_input


def de_emojify(text):
    regrex_pattern = re.compile(
        pattern="["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return regrex_pattern.sub(r"", text)


def print_verbose(verbose: bool, message: str):
    print(message) if verbose else None


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
    print_verbose(args.verbose, "Checking models folder...")
    try:
        os.listdir("models")
    except FileNotFoundError:
        print_verbose(args.verbose, "Creating models folder...")
        os.mkdir("models")

    with os.scandir("models") as folder:
        # Check for each model
        match args.intent_model:
            case "RF":
                print_verbose(args.verbose, "Using random forest model...")
                # If folder contains optimized_random_forest.joblib, then we are good to go
                if "optimized_random_forest.joblib" in [file.name for file in folder]:
                    return
                # Train model if not
                else:
                    print_verbose(args.verbose, "Training random forest...")
                    generate_random_forest()
                    print_verbose(args.verbose, "Done training random forest.")
                    return

            case "neural":
                print_verbose(args.verbose, "Using multi layer perceptron model...")
                # If folder contains mlp_model.pt, then we are good to go
                if "mlp_model.pt" in [file.name for file in folder]:
                    return
                # Train model if not
                else:
                    print_verbose(args.verbose, "Training multi layer perceptron...")
                    fit_mlp()
                    print_verbose(args.verbose, "Done training multi layer perceptron.")
                    return
            case "majority":
                print_verbose(args.verbose, "Using majority model...")
                return
            case "keyword":
                print_verbose(args.verbose, "Using keyword model...")
                return
            case _:
                raise ValueError(f"Invalid model: {args.intent_model}")


def add_properties():
    # get restaurant data
    information = pd.read_csv("data/restaurant_info.csv")

    # initialize lists
    food_quality = []
    crowdedness = []
    length_of_stay = []

    # for now these are the only values to choose from for all 3 new collumns.
    # Since these are for the inference model to use and not  for the user
    # these values are fairly basic to make the inference easier.
    food_quality_values = ["good", "bad"]
    crowdedness_values = ["busy", "calm"]
    length_of_stay_values = ["long", "short"]

    # create appropriatly sized new collumns
    for i in range(len(information["addr"])):
        food_quality.append(random.choice(food_quality_values))
        crowdedness.append(random.choice(crowdedness_values))
        length_of_stay.append(random.choice(length_of_stay_values))

    information["food_quality"] = food_quality
    information["crowdedness"] = crowdedness
    information["length_of_stay"] = length_of_stay

    # write to new csv file
    information.to_csv("data/restaurant_info_extra.csv")


def capitalize_first_letter(string):
    return string[0].upper() + string[1:]

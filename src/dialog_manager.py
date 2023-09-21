import random
from typing import TypedDict, list
from Levenshtein import distance
import joblib

from helpers import prep_user_input
from mlp_model.random_forest import predict_single_input


# DialogConfig type
class DialogConfig(TypedDict):
    intent_model: str  # Type of intent model, default is RandomForest
    verbose: bool  # Whether to print out debug information


class Message(TypedDict):
    classified_intent: str | None
    text: str
    sender: str


class DialogManager:
    """
    DialogManager class, which can be called to start a new conversation with the bot.
    """

    # -------------- Constructor --------------
    def __init__(self, dialog_config: DialogConfig):
        self.dialog_config = dialog_config
        self.done = False
        self.message_history: list[Message] = []
        self.message_templates = {
            "welcome": "Hello, I am a restaurant recommender chatbot \N{rocket}. How can I help you?",
            # "confirmfoodtype": f"",
        }
        self.stored_preferences = {
            "food_type": None,
            "price_range": None,
            "area": None,
        }
        self.options = ["Danish", "Spanish", "Italian"]
        self.intent_classifier = joblib.load("models/optimized_random_forest.joblib")

    # -------------- Interface methods --------------
    def __handle_input(self, user_input):
        # Process user input
        prepped_user_input = prep_user_input(user_input)
        self.__add_message(None, prepped_user_input, "user")

        # Check if user wants to exit
        if prepped_user_input == "exit":
            self.done = True
            return

        # Check user intent
        intent = self.__get_intent(prepped_user_input)

        self.__respond()

        # Check if user made a typo
        alternatives = self.get_levenshtein_alternatives(
            prepped_user_input, self.options
        )
        if alternatives:
            self.__respond("Did you mean one of the following?")
            self.show_matches(alternatives)
            return

    def __respond(self, input):
        print(f"\N{robot face} Bot: {input}")

    # -------------- Internal methods --------------
    def start_dialog(self):
        self.__respond(self.message_templates["welcome"])
        while not self.done:
            # Get the user input on the same line as the prompt
            print("\r\N{bust in silhouette} User: ", end="")
            user_input = input()
            self.__handle_input(user_input)

    def __get_intent(self, prepped_user_input):
        # Pre-process user input
        return predict_single_input(prepped_user_input)

    def __add_message(self, intent, text, sender):
        self.message_history.append(
            {
                "classified_intent": intent,
                "text": text,
                "sender": sender,
            }
        )

    # -------------- Helper methods --------------
    def get_levenshtein_alternatives(self, word, options):
        matches = []
        options_copy = options.copy()  # Copy options to avoid mutating original list
        options_copy = [
            prep_user_input(option) for option in options_copy
        ]  # Preprocess options just in case
        random.shuffle(options_copy)  # Shuffle options to avoid bias

        # Loop through options and calculate levenshtein distance
        for option in options_copy:
            dist = distance(word, option)
            # If distance is 0, then we have a perfect match
            if dist == 0:
                return None

            # If distance is less than 2, then we have a match
            if dist <= 2:
                matches.append(
                    {
                        "option": option,
                        "distance": dist,
                    }
                )

        # Sort matches by distance for easy handling
        matches.sort(key=lambda x: x["distance"])
        return matches

    def show_matches(self, matches):
        for match in matches:
            # Upper case first letter of option
            if match["option"] is not None:
                match["option"] = match["option"][0].upper() + match["option"][1:]
                print("\t- " + match["option"] + "?")

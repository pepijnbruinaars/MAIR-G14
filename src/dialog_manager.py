import random
from typing import TypedDict
from Levenshtein import distance
import joblib

from helpers import prep_user_input


# DialogConfig type
class DialogConfig(TypedDict):
    intent_model: str  # Type of intent model, default is RandomForest
    verbose: bool  # Whether to print out debug information


class DialogManager:
    """
    DialogManager class, which can be called to start a new conversation with the bot.
    """

    # -------------- Constructor --------------
    def __init__(self, dialog_config: DialogConfig):
        self.dialog_config = dialog_config
        self.done = False
        self.retrieved_info = {}
        self.message_templates = {
            "welcome": "Hello, I am a restaurant recommender chatbot \N{rocket}. How can I help you?",
            # "confirmfoodtype": f"",
        }
        self.options = ["Danish", "Spanish", "Italian"]
        self.intent_classifier = joblib.load("models/optimized_random_forest.joblib")

    # -------------- Interface methods --------------
    def __handle_input(self, user_input):
        prepped_user_input = prep_user_input(user_input)
        if prepped_user_input == "exit":
            self.done = True
            return

        self.__respond("Intent: " + self.__get_intent(prepped_user_input))
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
            user_input = input()  # Get user input
            # print("\033[A \033[A")  # Clear user input
            self.__handle_input(user_input)

    def __get_intent(self, prepped_user_input):
        # Pre-process user input
        return (prepped_user_input)[0]

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
            if dist <= 2:
                # Store option and distance in dict
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
                print(match["option"])

import random
from typing import TypedDict
from Levenshtein import distance
import joblib
import pandas as pd
import re

from helpers import prep_user_input
from mlp_model.random_forest import predict_single_input

information = pd.read_csv("data/restaurant_info.csv")
    

# Enum for all intent types
class IntentType:
    ACK = "acknowledgment"
    AFFIRM = "affirm"
    BYE = "bye"
    CONFIRM = "confirm"
    DENY = "deny"
    HELLO = "hello"
    INFORM = "inform"
    NEGATE = "negate"
    NULL = "null"
    REPEAT = "repeat"
    REQALTS = "reqalts"
    REQMORE = "reqmore"
    REQUEST = "request"
    RESTART = "restart"
    THANKYOU = "thankyou"


# DialogConfig type
class DialogConfig(TypedDict):
    intent_model: str  # Type of intent model, default is RandomForest
    verbose: bool  # Whether to print out debug information


class Message(TypedDict):
    classified_intent: IntentType
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
        self.options = ['british', 'modern european', 'italian', 'romanian', 'seafood',
           'chinese', 'steakhouse', 'asian oriental', 'french', 'portuguese',
           'indian', 'spanish', 'european', 'vietnamese', 'korean', 'thai',
           'moroccan', 'swiss', 'fusion', 'gastropub', 'tuscan',
           'international', 'traditional', 'mediterranean', 'polynesian',
           'african', 'turkish', 'bistro', 'north american', 'australasian',
           'persian', 'jamaican', 'lebanese', 'cuban', 'japanese', 'catalan',
           'moderate', 'expensive', 'cheap','west', 'north', 'south', 'centre', 
           'east']  
        self.intent_classifier = joblib.load("models/optimized_random_forest.joblib")

    def __repr__(self):
        return f"DialogManager({self.dialog_config})"

    # -------------- Interface methods --------------
    def __handle_input(self, user_input):
        # Process user input
        prepped_user_input = prep_user_input(user_input)

        # Check user intent
        intent = self.__get_intent(prepped_user_input)
        self.__add_message(intent, prepped_user_input, "user")
        
        # extract the prefences for a restaurant the user might have uttered 
        self.__extract_preference(prepped_user_input)
        

        # Check if user wants to exit
        if prepped_user_input == "exit":
            self.__handle_exit()
            return

        # Logging for debugging
        if self.dialog_config["verbose"]:
            print(f"Intent: {intent}")
            print(f"User input: {prepped_user_input}")

        self.__respond(f"Your intent is {intent}?")

        

    def __respond(self, input):
        self.__add_message(None, input, "bot")
        print(f"\N{robot face} Bot: {input}")

    def __print_message_history(self):
        for message in self.message_history:
            print(message)

    def __handle_exit(self):
        self.__respond("Goodbye! \N{waving hand sign}")
        if self.dialog_config["verbose"]:
            self.__print_message_history()
        self.done = True

    # -------------- Public methods --------------
    def start_dialog(self):
        self.__respond(self.message_templates["welcome"])
        while not self.done:
            # Get the user input on the same line as the prompt
            print("\r\N{bust in silhouette} User: ", end="")
            user_input = input()
            self.__handle_input(user_input)

    # -------------- Internal methods --------------
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
    def __get_levenshtein_alternatives(self, word, options):
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
                print( f"we went looking for closest word but found the correct word: {option}")
                return
            
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

    def __show_matches(self, matches):
        for match in matches:
            # Upper case first letter of option
            if match["option"] is not None:
                match["option"] = match["option"][0].upper() + match["option"][1:]
                print("\t- " + match["option"] + "?")

    
    
    def __extract_preference(self, input_string : str):
        
        # make sure input is in lower case
        input_string = input_string.lower()
        
        # First entry
        food_regex = ""
        
        # for every other entry add the option of food
        for i in self.options[:-8]:
            # First entry is british, doesn't need an or prefix ( | )
            if i == "british":
                food_regex = i
            else:
                food_regex = food_regex + "|" + i 
        
        # match the possible preferences to the input
        area_match = re.search(r"west|north|south|centre|east", input_string)
        food_match = re.search(r"moderate|expensive|cheap", input_string)
        price_match = re.search(rf"{food_regex}", input_string)
    
        found_something = False
        if food_match:
            self.stored_preferences["food"] = food_match.group()
            found_something = True
            
        if area_match:
            self.stored_preferences["area"] = area_match.group()
            found_something = True

        if price_match:
            self.stored_preferences["price_range"] = price_match.group()
            found_something = True
            
        if not found_something:
            if self.dialog_config["verbose"]:
                print("no preference found")
                
            # find closest with levenshtein distance (max = 3)
            for i in input_string.split(" "):
                matches = self.__get_levenshtein_alternatives(i, self.options)
                if matches:
                    self.__respond("Did you mean one of the following?")
                    self.__show_matches(matches)
                
        return
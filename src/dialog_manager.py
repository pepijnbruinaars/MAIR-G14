import random
from typing import TypedDict
from Levenshtein import distance

from helpers import prep_user_input

# DialogConfig type
class DialogConfig(TypedDict):
    intent_model: str # Type of intent model, default is RandomForest
    verbose: bool # Whether to print out debug information

class DialogManager():
    # Init
    def __init__(self, dialog_config):
        self.dialog_config = dialog_config
        self.done = False
        self.retrieved_info = {}
        self.message_templates = {
            "welcome": "Hello, I am a restaurant recommender chatbot. How can I help you?",
            "confirmfoodtype": f""
        }
        self.options = ['Danish', 'Spanish', 'Italian']
    
    # TODO: Load intent model
    def load_intent_model(self):
        pass
    
    # Handle user input
    def __handle_input(self, user_input):
        user_input = prep_user_input(user_input)
        if user_input == "exit":
            self.done = True
            return
        
        print("You said: " + user_input)
        alternatives = self.get_levenshtein_alternatives(user_input, self.options)
        print(alternatives)
        if alternatives:
            print("Did you mean one of the following?")
            self.show_matches(alternatives)
            return
    
    # Intialize dialog
    def start_dialog(self):
        print(self.message_templates["welcome"])
        while not self.done:
            user_input = input()
            self.__handle_input(user_input)

            
    def get_levenshtein_alternatives(self, word, options):
        matches = []
        options_copy = options.copy() # Copy options to avoid mutating original list
        options_copy = [prep_user_input(option) for option in options_copy] # Preprocess options just in case
        random.shuffle(options_copy) # Shuffle options to avoid bias
        
        # Loop through options and calculate levenshtein distance
        for option in options_copy:
            dist = distance(word, option)
            if dist <= 2:
                # Store option and distance in dict
                matches.append({
                    'option': option, 
                    'distance': dist,
                })
                
        # Sort matches by distance for easy handling
        matches.sort(key=lambda x: x["distance"])
        return matches
    
    def show_matches(self, matches):
        for match in matches:
            # Upper case first letter of option
            if match["option"] is not None:
                match["option"] = match["option"][0].upper() + match["option"][1:]
                print(match["option"])
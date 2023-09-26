from intent_models.ml_models.random_forest import predict_single_input
from intent_models.baselines.keyword_matching import match_sentence
from helpers import prep_user_input
from Levenshtein import distance
from typing import TypedDict
from textwrap import dedent
from pygame import mixer
from io import BytesIO
from gtts import gTTS
import pandas as pd
import random
import re

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
    intent_model: str   # Type of intent model, default is RandomForest
    verbose: bool       # Whether to print out debug information
    tts: bool           # Wheter to convert the system output to speech
    caps: bool          # Wheter to print the system output in all caps
    levenshtein: int    # Integer defining the desired levenshtein distance
    delay: float        # Optional delay before the system responds


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
            "food": None,
            "pricerange": None,
            "area": None,
        }
        self.food_options = information["food"].unique()
        self.price_options = information["pricerange"].unique()
        self.area_options = ["west", "north", "south", "centre", "east"]

        self.options = [
            "Danish",
            "Spanish",
            "Italian",
            "phone",
            "number",
            "telephone",
            "contact",
            "address",
            "located",
            "location",
            "where",
            "postcode",
            "post code",
            "post",
            "code",
        ]

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

        # Retrieve restaurant based on preferences
        restaurant, other_options = self.__retrieve_restaurant(self.stored_preferences)

        # Handle user intent
        match intent:
            case IntentType.ACK:
                self.__respond("You're welcome!")
            case IntentType.AFFIRM:
                self.__respond("Great!")
            case IntentType.BYE:
                self.__handle_exit()
            case IntentType.INFORM:
                self.__handle_inform(restaurant)
            case IntentType.REQUEST:
                if restaurant is not None:
                    self.__handle_request(prepped_user_input, restaurant)
            case IntentType.RESTART:
                self.stored_preferences = {
                    "food": None,
                    "pricerange": None,
                    "area": None,
                }
            case _:  # Default case
                self.__respond("I'm sorry, I don't understand.")

    def __respond(self, input):
        self.__add_message(None, input, "bot")
        print(f"\N{robot face} Bot: {input}")
        if self.dialog_config["tts"]:
            mp3_fp = BytesIO()
            tts = gTTS(input, lang='en', tld="com")
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)

            mixer.init()
            mp3_fp.seek(0)
            mixer.music.load(mp3_fp, "mp3")
            mixer.music.play()

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
        if self.dialog_config["intent_model"] == "RF":
            return predict_single_input(prepped_user_input)
        if self.dialog_config["intent_model"] == "keyword":
            return match_sentence(prepped_user_input)
        if self.dialog_config["intent_model"] == "majority":
            return "inform"  # This is the majority class

    def __add_message(self, intent, text, sender):
        self.message_history.append(
            {
                "classified_intent": intent,
                "text": text,
                "sender": sender,
            }
        )

    def __handle_inform(self, restaurant) -> bool:
        self.__respond(self.__get_suggestion_string(restaurant))

    def __handle_request(self, prepped_user_input, restaurant) -> bool:
        # in findoutuserintent, checks for phone, addr and postcode and returns it

        output = ""

        phone_matches = ["phone", "number", "telephone", "contact"]
        address_matches = ["address", "located", "location", "where"]
        postcode_matches = ["postcode", "post code", "post", "code"]

        for match in phone_matches:
            # check for phone number
            if match in prepped_user_input:
                output = f"The phone number of the restaurant is {restaurant['phone']}."
                break

        for match in address_matches:
            # check for address
            if match in prepped_user_input:
                if len(output) == 0:
                    output = f"The {restaurant['name']} is on {restaurant['addr']}. "
                else:
                    # respond differently if phone number is asked as well
                    output = (
                        output[:-1] + f" and it is located on {restaurant['addr']}."
                    )
                break

        for match in postcode_matches:
            # check for postcode
            if match in prepped_user_input:
                if len(output) == 0:
                    output = f"The post code of {restaurant['name']} is {restaurant['postcode']}."
                else:
                    # respond differently if phone number or address is asked as well
                    output = (
                        output[:-1] + f", its postcode is {restaurant['postcode']}."
                    )
                break

        # if no information was found return to findoutuserintent
        if len(output) == 0:
            return False

        # else provide the user with the information
        self.__respond(self, output)

        return True

    # -------------- Helper methods --------------
    def __get_levenshtein_alternatives(self, word, options) -> list[dict]:
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

    def __show_matches(self, matches) -> None:
        for match in matches:
            # Upper case first letter of option
            if match["option"] is not None:
                match["option"] = match["option"][0].upper() + match["option"][1:]
                print("\t- " + match["option"] + "?")

    def __extract_preference(self, input_string: str) -> None:
        # make sure input is in lower case
        input_string = input_string.lower()

        # for every entry add the option of to the regex
        food_regex = "|".join(self.food_options)
        area_regex = "|".join(self.area_options)
        price_regex = "|".join(self.price_options)

        # match the possible preferences to the input
        food_match = re.search(rf"{food_regex}", input_string)
        area_match = re.search(rf"{area_regex}", input_string)
        price_match = re.search(rf"{price_regex}", input_string)

        # If we find something, we don't need to look for something mistyped anymore
        # Look for exact matches
        found_something = False
        if food_match:
            self.stored_preferences["food"] = food_match.group()
            found_something = True

        if area_match:
            self.stored_preferences["area"] = area_match.group()
            found_something = True

        if price_match:
            self.stored_preferences["pricerange"] = price_match.group()
            found_something = True

        if not found_something:
            if self.dialog_config["verbose"]:
                print("no preference found")

            # concat all options to look for mistyped ones
            all_options = np.concatenate(
                (self.food_options, self.area_options, self.price_options)
            )

            # find closest with levenshtein distance (max = 3)
            for i in input_string.split(" "):
                matches = self.__get_levenshtein_alternatives(i, all_options)
                if matches:
                    self.__respond("Did you mean one of the following?")
                    self.__show_matches(matches)

        # Debug information
        if self.dialog_config["verbose"]:
            print("extracted type preference: ", self.stored_preferences["food"])
            print("extracted area preference: ", self.stored_preferences["area"])
            print("extracted price preference: ", self.stored_preferences["pricerange"])

        return

    def __retrieve_restaurant(self, preferences):
        """Function which retrieves a restaurant based on the user's preferences

        Args:
            preferences (_type_): The retrieved preferences of the user
        """
        data = pd.read_csv("data/original/restaurant_info.csv")
        pref_type = preferences["food"]
        pref_area = preferences["area"]
        pref_price = preferences["pricerange"]

        restaurant_choice = None
        other_options = None

        # If no preferences are given, return None
        if pref_type is None and pref_area is None and pref_price is None:
            return None, None

        if pref_type is not None:
            data = data[data["food"] == pref_type]
        if pref_area is not None:
            data = data[data["area"] == pref_area]
        if pref_price is not None:
            data = data[data["pricerange"] == pref_price]

        if len(data) == 1:
            restaurant_choice = data
        elif len(data) > 1:
            restaurant_choice = data.sample(n=1)
            restaurant_choice_name = restaurant_choice["restaurantname"].iloc[0]
            other_options = data[data["restaurantname"] != restaurant_choice_name]

        # Create dict with restaurant information
        if restaurant_choice is not None:
            restaurant_choice = restaurant_choice.to_dict("records")[0]
        if other_options is not None:
            other_options = other_options.to_dict("records")

        return restaurant_choice, other_options

    def __get_suggestion_string(self, restaurant):
        """Function which returns a string with a restaurant suggestion

        Args:
            restaurant (_type_): The retrieved restaurant
        """
        if restaurant is not None:
            # Remove new lines from the returned string
            return dedent(
                f"""\
                I suggest you go to {restaurant['restaurantname']}. It's {self.__get_word_prefix(restaurant['food'])}
                {restaurant['food']} restaurant in the {restaurant['area']} of town."""
            ).replace("\n", " ")
        return "I'm sorry, I don't know any restaurants that match your preferences."

    def __get_word_prefix(self, word):
        """Function which a or an based on the first letter of a word

        Args:
            word (_type_): The word to get the prefix of
        """
        return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"

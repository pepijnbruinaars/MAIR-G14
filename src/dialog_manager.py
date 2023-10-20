import os
from helpers import (
    capitalize_first_letter,
    get_message_templates,
    get_suggestion_prefixes,
    prep_user_input,
    de_emojify,
    print_verbose,
    get_identity,
)  # noqa

# Necessary to hide the pygame import message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from intent_models.ml_models.random_forest import predict_single_input_rf  # noqa
from intent_models.baselines.keyword_matching import match_sentence  # noqa
from intent_models.ml_models.mlp import predict_single_input_mlp  # noqa
from voice_model import text_to_speech, Speakers  # noqa
from Levenshtein import distance  # noqa
from typing import TypedDict  # noqa
from textwrap import dedent  # noqa
from io import BytesIO  # noqa
from gtts import gTTS  # noqa

import speech_recognition as sr  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa
import random  # noqa
import pygame  # noqa
import time  # noqa
import re  # noqa

information = pd.read_csv("data/restaurant_info_extra.csv")


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
    tts: bool  # Whether to convert the system output to speech
    caps: bool  # Whether to print the system output in all caps
    levenshtein: int  # Integer defining the desired levenshtein distance
    delay: float  # Optional delay before the system responds
    speech: bool  # Whether to take user input as speech or not
    typing_speed: float  # Optional typing speed multiplier for the system
    gender: str


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
        self.message_templates = get_message_templates()
        self.message_history: list[Message] = []
        self.emoji, self.name = get_identity(dialog_config["gender"])

        self.stored_preferences = {
            "food": None,
            "pricerange": None,
            "area": None,
        }

        # Stored restaurant information
        self.stored_restaurant = None
        self.stored_restaurant_options = None

        # Preference options
        self.food_options = information["food"].unique().tolist()
        self.price_options = information["pricerange"].unique().tolist()
        self.area_options = ["west", "north", "south", "centre", "east"]
        self.additional_query = False
        self.has_given_recommendation = False

    def __repr__(self):
        return f"DialogManager({self.dialog_config})"

    # -------------- Interface methods --------------
    def __handle_input(self, user_input):
        # Declare global raw input to use in extracting preference
        global raw_input
        raw_input = user_input

        # Process user input
        prepped_user_input = prep_user_input(user_input)

        # Check if user wants to exit
        if prepped_user_input in ["exit", "quit", "bye", "goodbye"]:
            self.__add_message("exit", prepped_user_input, "User")
            self.__handle_exit()
            return

        # Check user intent
        intent = self.__get_intent(prepped_user_input)
        self.__add_message(intent, prepped_user_input, "User")

        # Logging for debugging
        print_verbose(self.dialog_config["verbose"], f"Intent: {intent}")
        print_verbose(
            self.dialog_config["verbose"], f"User input: {prepped_user_input}"
        )

        # Handle user intent
        print_verbose(self.dialog_config["verbose"], self.stored_restaurant)
        print_verbose(self.dialog_config["verbose"], self.stored_restaurant_options)
        print_verbose(self.dialog_config["verbose"], self.stored_preferences)
        if (
            self.stored_restaurant is not None
            and self.stored_restaurant_options is None
            and self.has_given_recommendation
            and (
                intent != IntentType.REQALTS
                and intent != IntentType.REQMORE
                and intent != IntentType.REQUEST
                and intent != IntentType.RESTART
            )
        ) or (
            self.stored_restaurant is not None
            and self.stored_restaurant_options is not None
            and all(self.stored_preferences.values())
            and self.has_given_recommendation
            and (
                intent != IntentType.REQALTS
                and intent != IntentType.REQMORE
                and intent != IntentType.REQUEST
                and intent != IntentType.RESTART
            )
        ):
            self.__extract_and_handle_additional_preferences(prepped_user_input)
        else:
            self.__handle_intents(intent, prepped_user_input)

    def __respond(self, response: str):
        # Handle delay
        self.__handle_delay() if self.dialog_config[
            "delay"
        ] > 0.0 and not self.dialog_config["verbose"] else None

        # Handle caps
        response = response.upper() if self.dialog_config["caps"] else response

        # Handle text to speech
        if self.dialog_config["tts"]:
            self.__handle_tts(response)
            return

        # Add message to history and display
        self.__add_message(None, response, self.name)
        # Show response word for word to simulate typing
        if not self.dialog_config["verbose"]:
            print(f"\r{self.emoji} {self.name}: ", end="")
            for c in response:
                print(c, end="", flush=True),
                if self.dialog_config["typing_speed"] != 0:
                    time.sleep(
                        1
                        / self.dialog_config["typing_speed"]
                        * random.uniform(0.005, 0.08)
                    ),
            print()
        else:
            print("\r{} {}: {}\n".format(self.emoji, self.name, response), end="")

    def __print_message_history(self):
        if self.dialog_config["verbose"]:
            print("\n------------- Message history -------------")
            for message in self.message_history:
                emoji = (
                    self.emoji
                    if message["sender"].lower() == self.name.lower()
                    else "\N{bust in silhouette}"
                )
                if message["classified_intent"]:
                    print(
                        f"{emoji} {message['sender']}: {message['text']} ({message['classified_intent']})"
                    )
                else:
                    print(f"{emoji} {message['sender']}: {message['text']}")

            print("-------------- End of dialog -------------")

    def __handle_exit(self):
        self.__respond("Goodbye! \N{waving hand sign}")
        # If mixer is still playing do not exit yet
        if pygame.mixer.get_init() is not None:
            # Wait until voice is done with going to the next sentence
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        self.__print_message_history()
        self.done = True

    def __handle_delay(self):
        start_time = time.time()
        counter = 1
        while time.time() - start_time < self.dialog_config["delay"]:
            if counter > 3:
                print(f"{self.emoji} {self.name}: {' ' * counter}", end="\r")
                counter = 0
            print(f"{self.emoji} {self.name}: {'.' * counter}", end="\r")
            counter += 1
            time.sleep(0.1)

    # -------------- Public methods --------------
    def start_dialog(self):
        _, name = get_identity(self.dialog_config["gender"])
        self.__respond(
            self.message_templates["welcome"].replace("I am", f"I am {name},")
            + "\n"
            + "\tWhat can I do for you?"
        )
        try:
            self.__dialog_loop()
        except KeyboardInterrupt:
            print()
            self.__handle_exit()

    # -------------- Internal methods --------------
    def __dialog_loop(self):
        while not self.done:
            # Get the user input on the same line as the prompt
            print("\N{bust in silhouette} User: ", end="")
            user_input = (
                self.__handle_speech() if self.dialog_config["speech"] else input()
            )

            self.__handle_input(user_input)

    def __get_intent(self, prepped_user_input):
        match self.dialog_config["intent_model"]:
            case "RF":
                return predict_single_input_rf(prepped_user_input)
            case "neural":
                return predict_single_input_mlp(prepped_user_input)
            case "keyword":
                return match_sentence(prepped_user_input)
            case "majority":
                return "inform"  # This is the majority class

    def __handle_intents(self, intent, prepped_user_input):
        match intent:
            case IntentType.ACK:
                # Handle acknowledgement, which isn't very informative
                self.__respond("Okay!")
            case IntentType.AFFIRM:
                # Handle affirmation, which isn't very informative either
                self.__respond("Great!")
            case IntentType.BYE:
                # Exit dialog system
                self.__handle_exit()
            case IntentType.INFORM:
                # Handle inform
                self.__handle_inform(prepped_user_input)
            case IntentType.HELLO:
                # Generic hello message
                self.__respond(self.message_templates["hello"])
            case IntentType.THANKYOU:
                # Generic thank you message
                self.__respond(self.message_templates["thankyou"])
            case IntentType.NEGATE | IntentType.DENY:
                # Handle deny and negation in the same way
                self.__handle_negate()
            case IntentType.REQUEST:
                # We can only handle requests if we have a restaurant
                if self.stored_restaurant is not None:
                    self.__handle_request(prepped_user_input, self.stored_restaurant)
                else:
                    self.__respond(self.message_templates["err_req"])
            case IntentType.REQMORE:
                # We can only handle requests if we have a restaurant and other options
                if (
                    self.stored_restaurant is not None
                    and self.stored_restaurant_options is not None
                ):
                    self.__respond("Here are some other options:")
                    self.__show_matches(self.stored_restaurant_options)
                else:
                    self.__respond(self.message_templates["err_req"])
            case IntentType.REPEAT:
                # Just respond the latest message sent by the bot again
                # Find the last message with sender bot
                last_bot_message = None
                for message in reversed(self.message_history):
                    if message["sender"] == "Bot":
                        last_bot_message = message
                        break
                # If we found a message, repeat it
                if last_bot_message is not None:
                    self.__respond(last_bot_message["text"])
                else:
                    self.__respond("I'm sorry, I don't have anything to repeat.")
            case IntentType.RESTART:
                # Reset preferences and stored restaurant information
                self.stored_restaurant = None
                self.stored_restaurant_options = None
                self.stored_preferences = {
                    "food": None,
                    "pricerange": None,
                    "area": None,
                }
                self.__respond(
                    "Your preferences have been reset! What can I do for you?"
                )
            case IntentType.REQALTS:
                # Handle request for alternatives
                self.__handle_reqalts(prepped_user_input)
            case IntentType.CONFIRM:
                # Handle a confirmation request
                self.__handle_confirm(prepped_user_input)
            case _:  # Default (null) case
                self.__respond("I'm sorry \N{pensive face}, I don't understand.")

    def __add_message(self, intent, text, sender):
        self.message_history.append(
            {
                "classified_intent": intent,
                "text": text,
                "sender": sender,
            }
        )

    # -------------- Intent handling methods --------------
    def __handle_inform(self, prepped_user_input) -> bool:
        # extract the prefences for a restaurant the user might have uttered
        self.__extract_preference(prepped_user_input)

        # Update restaurant information
        restaurant, other_options = self.__retrieve_restaurant(self.stored_preferences)
        self.stored_restaurant = restaurant
        self.stored_restaurant_options = other_options

        # If no options left, and no restaurant found, then we can't help the user
        if restaurant is None and other_options is None:
            self.__respond(self.message_templates["err_inf_no_result"])
            return False

        # If 1 option left, suggest it, we also cannot ask the user for additional preferences
        if restaurant is not None and other_options is None:
            self.__respond(self.__get_suggestion_string(restaurant))
            self.__prompt_additional_preferences()
            self.has_given_recommendation = True
            return True

        # If more than 1 option left, and all preferences are filled, suggest one and ask for additional preferences
        if (
            restaurant is not None
            and other_options is not None
            and all(self.stored_preferences.values())
        ):
            self.__prompt_other_preferences()  # Prompt for other required preferences
            if not self.additional_query:
                self.__respond(self.__get_suggestion_string(restaurant))
                self.__prompt_additional_preferences()
                self.has_given_recommendation = True
                return True

            return False

        # If more than 1 option left, and not all preferences are filled, ask for more preferences
        if (
            restaurant is not None
            and other_options is not None
            and not all(self.stored_preferences.values())
        ):
            self.__prompt_other_preferences()
            return True
        return False

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
                    output = (
                        f"{restaurant['restaurantname']} is on {restaurant['addr']}. "
                    )
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
                    output = f"The post code of {restaurant['restaurantname']} is {restaurant['postcode']}."
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
        self.__respond(output)
        self.__respond("Is there anything else I can help you with?")

        return True

    def __handle_negate(self):
        # If we have a restaurant, then we can suggest other options
        if self.stored_restaurant is not None:
            # If we have other options, suggest one
            if self.stored_restaurant_options is not None:
                self.__respond("Here are some other options:")
                self.__show_matches(self.stored_restaurant_options)
            # If we don't have other options, then we can't help the user
            else:
                self.__respond(self.message_templates["err_neg_no_options"])
        # If we don't have a restaurant, then we can't help the user
        else:
            self.__respond(self.message_templates["err_neg_next_step"])

    def __handle_reqalts(self, prepped_user_input):
        self.__extract_preference(prepped_user_input)
        previous_suggestion = self.stored_restaurant

        # Update restaurant information
        restaurant, other_options = self.__retrieve_restaurant(self.stored_preferences)
        self.stored_restaurant = restaurant
        self.stored_restaurant_options = other_options

        # If no options left, and no restaurant found, then we can't help the user
        if restaurant is None and other_options is None:
            self.__respond(self.message_templates["err_inf_no_result"])
            return False

        # If 1 option left, suggest it
        if (
            restaurant is not None
            and restaurant["restaurantname"] != previous_suggestion["restaurantname"]
            and other_options is None
        ):
            self.__respond(self.__get_suggestion_string(restaurant))
            return True

        # If more than 1 option left, and all preferences are filled, suggest one
        if (
            restaurant is not None
            and other_options is not None
            and all(self.stored_preferences.values())
        ):
            self.__prompt_other_preferences()
            if restaurant != previous_suggestion:
                self.__respond(self.__get_suggestion_string(restaurant))
                return True

            self.__respond("There are no other options that match your preferences.")
            self.__respond(self.__get_suggestion_string(restaurant))

            return True
        self.__respond("There are no other options that match your preferences.")
        self.__respond(self.__get_suggestion_string(restaurant))
        return False

    def __handle_confirm(self, prepped_user_input):
        preferences = self.__extract_preference(prepped_user_input, False)
        chosen_restaurant = self.stored_restaurant

        # If no preferences are given, return None
        if (
            preferences["food"] is None
            and preferences["area"] is None
            and preferences["pricerange"] is None
        ) or chosen_restaurant is None:
            self.__respond(
                "I haven't found a restaurant yet so I can't help you with that..."
            )

            return False

        # Check if preferences are satisfied
        food_match = preferences["food"] == chosen_restaurant["food"]
        area_match = preferences["area"] == chosen_restaurant["area"]
        price_match = preferences["pricerange"] == chosen_restaurant["pricerange"]

        # If all preferences are satisfied, confirm
        if food_match and area_match and price_match:
            self.__respond("Yes, that's correct!")
            return True
        if not food_match:
            self.__respond(
                f"Nope, the food is {capitalize_first_letter(chosen_restaurant['food'])}."
            )
        if not area_match:
            self.__respond(f"Nope, the area is {chosen_restaurant['area']}.")

        if not price_match:
            self.__respond(
                f"Nope, the price range is {chosen_restaurant['pricerange']}."
            )

        return False

    # -------------- Speech methods --------------
    def __handle_tts(self, response: str):
        # Add message to history and display
        self.__add_message(None, response, self.name)

        # Get speaker
        speaker = (
            Speakers.FEMALE
            if self.dialog_config["gender"] == "female"
            else Speakers.MALE
        )

        # Keep original sentences for printing but in split format
        org_sentences = [
            sentence
            for sentence in re.split("(?<=[!?.\r\n])", response)
            if sentence.lstrip() not in ["\r", "\n", ""]
        ]

        # prepare sentences for voice synthesization
        sentences = [
            sentence.lstrip().replace("\n", "").__repr__().replace("'", "")
            for sentence in re.split("(?<=[!?.\r\n])", de_emojify(response))
            if sentence.lstrip() != ""
        ]

        # Print and generate voice
        for index, (text, voice_text) in enumerate(zip(org_sentences, sentences)):
            text_to_speech(voice_text, speaker)

            prefix = f"{self.emoji} {self.name}: " if index == 0 else " " * (len(self.name) + 4)
            print(prefix, end="")

            # Show text word for word to simulate typing
            if not self.dialog_config["verbose"]:
                for c in text:
                    print(c, end="", flush=True),
                    if self.dialog_config["typing_speed"] != 0:
                        time.sleep(
                            1
                            / self.dialog_config["typing_speed"]
                            * random.uniform(0.005, 0.08)
                        ),
                print() if "\n" not in text else print("", end="")
            else:
                print(text, end="")
                print() if "\n" not in text else print("", end="")

    def __handle_speech(self):
        recognizer = sr.Recognizer()

        # Capture audio from the microphone
        with sr.Microphone() as source:
            audio = recognizer.listen(
                source, timeout=None, phrase_time_limit=5
            )  # Adjust the timeout as needed
        try:
            # Recognize the audio using Google Web Speech API
            user_input = recognizer.recognize_google(audio)
            # Write user input letter for letter
            [(print(c, end="", flush=True), time.sleep(0.02)) for c in user_input]
            print()

            return user_input

        except sr.UnknownValueError:
            self.__respond("Sorry, I couldn't understand what you said.")
            print("\r\N{bust in silhouette} User: ", end="")
            return self.__handle_speech()
        except sr.RequestError as e:
            print_verbose(f"Sorry, an error occurred: {e}")

    # -------------- Preference and lookup methods --------------
    def __extract_preference(
        self, input_string: str, updatePreference: bool = True
    ) -> dict:
        preferences = {}

        # for every entry add the option of to the regex
        food_regex = "|".join(self.food_options)
        area_regex = "|".join(self.area_options)
        price_regex = "|".join(self.price_options)

        # match the possible preferences to the input
        food_match = re.search(rf"{food_regex}", input_string)
        area_match = re.search(rf"{area_regex}", input_string)
        price_match = re.search(rf"{price_regex}", input_string)

        any_match_food = re.search(
            r"(any|all|does not matter|dont care|regardless|whatever).*(food|restaurant|kitchen)",
            raw_input.lower(),
        )
        any_match_area = re.search(
            r"(any|all|does not matter|dont care|regardless|whatever).*(area|town|where)",
            raw_input.lower(),
        )
        any_match_price = re.search(
            r"(any|all|does not matter|dont care|regardless|whatever).*(price|cost)",
            raw_input.lower(),
        )

        print_verbose(self.dialog_config["verbose"], f"Raw input: {raw_input}")
        # If we find something, we don't need to look for something mistyped anymore
        # Look for exact matches
        found_something = False
        if food_match:
            preferences["food"] = food_match.group()
            found_something = True

        if area_match:
            preferences["area"] = area_match.group()
            found_something = True

        if price_match:
            preferences["pricerange"] = price_match.group()
            found_something = True

        if any_match_food:
            preferences["food"] = "any"
        if any_match_area:
            preferences["area"] = "any"
        if any_match_price:
            preferences["pricerange"] = "any"

        if updatePreference:
            self.stored_preferences.update(preferences)

        if not found_something:
            print_verbose(self.dialog_config["verbose"], "No exact matches found.")

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
        print_verbose(
            self.dialog_config["verbose"],
            f"extracted type preference: {self.stored_preferences['food']}",
        )
        print_verbose(
            self.dialog_config["verbose"],
            f"extracted area preference: {self.stored_preferences['area']}",
        )
        print_verbose(
            self.dialog_config["verbose"],
            f"extracted price preference: {self.stored_preferences['pricerange']}",
        )

        return preferences

    def __extract_additional_preferences(self, input_string: str) -> dict:
        requirements = {
            "touristic": None,
            "assigned_seats": None,
            "children": None,
            "romantic": None,
        }

        # for every entry add the possible preferences to the regex
        touristic_regex = "|".join(["touristic", "touristy", "tourism"])
        not_touristic_regex = "|".join(["not touristic", "not touristy", "not tourism"])
        assigned_seats_regex = "|".join(["assigned seats", "assigned seat"])
        not_assigned_seats_regex = "|".join(["not assigned seats", "not assigned seat"])
        children_regex = "|".join(["children", "child", "kids", "kid"])
        not_children_regex = "|".join(
            ["not children", "not child", "not kids", "no kids"]
        )
        romantic_regex = "|".join(["romantic", "romance", "romanticism"])
        not_romantic_regex = "|".join(
            ["not romantic", "not romance", "not romanticism"]
        )

        # match the possible preferences to the input
        touristic_match = re.search(rf"{touristic_regex}", input_string)
        not_touristic_match = re.search(rf"{not_touristic_regex}", input_string)
        assigned_seats_match = re.search(rf"{assigned_seats_regex}", input_string)
        not_assigned_seats_match = re.search(
            rf"{not_assigned_seats_regex}", input_string
        )
        children_match = re.search(rf"{children_regex}", input_string)
        not_children_match = re.search(rf"{not_children_regex}", input_string)
        romantic_match = re.search(rf"{romantic_regex}", input_string)
        not_romantic_match = re.search(rf"{not_romantic_regex}", input_string)

        # We don't have to look for anything mistyped anymore
        # Look for exact matches
        found_something = False
        if touristic_match:
            requirements["touristic"] = True
            found_something = True
        if not_touristic_match:
            requirements["touristic"] = False
            found_something = True
        if assigned_seats_match:
            requirements["assigned_seats"] = True
            found_something = True
        if not_assigned_seats_match:
            requirements["assigned_seats"] = False
            found_something = True
        if children_match:
            requirements["children"] = True
            found_something = True
        if not_children_match:
            requirements["children"] = False
            found_something = True
        if romantic_match:
            requirements["romantic"] = True
            found_something = True
        if not_romantic_match:
            requirements["romantic"] = False
            found_something = True

        if not found_something:
            self.__respond("I'm sorry, I don't understand.")
            return None

        return requirements

    def __extract_and_handle_additional_preferences(self, input_string: str) -> dict:
        # Extract additional preferences
        additional_preferences = self.__extract_additional_preferences(input_string)
        if additional_preferences is None:
            return True

        # Filter out restaurants that don't match the additional preferences
        if not self.stored_restaurant_options:
            self.stored_restaurant_options = []
        restaurants = [self.stored_restaurant] + self.stored_restaurant_options
        restaurant, other_options, reasons = self.__additional_preferences(
            restaurants, additional_preferences
        )

        # If no options left, but we do have a restaurant, we suggest
        if restaurant is not None:
            reason = reasons["restaurantname" == restaurant["restaurantname"]]
            self.__respond(self.__get_suggestion_string(restaurant))
            self.__respond(
                "That is because: {}"
                #    Format reason with every key in the reason dict except restaurantname
                .format(
                    ", ".join(
                        [
                            f"{key} {reason[key]}"
                            for key in reason
                            if key != "restaurantname"
                        ]
                    )
                )
            )
            return True

        self.__respond(self.message_templates["err_inf_no_result"])
        return False

    def __retrieve_restaurant(self, preferences):
        """Function which retrieves a restaurant based on the user's preferences

        Args:
            preferences (_type_): The retrieved preferences of the user
        """
        data = pd.read_csv("data/restaurant_info_extra.csv")
        pref_type = preferences["food"]
        pref_area = preferences["area"]
        pref_price = preferences["pricerange"]

        restaurant_choice = None
        other_options = None

        # If no preferences are given, return None
        if pref_type is None and pref_area is None and pref_price is None:
            return None, None

        if pref_type is not None and pref_type != "any":
            data = data[data["food"] == pref_type]
        if pref_area is not None and pref_area != "any":
            data = data[data["area"] == pref_area]
        if pref_price is not None and pref_price != "any":
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

        print_verbose(self.dialog_config["verbose"], "---- First choice ----")
        print_verbose(
            self.dialog_config["verbose"],
            self.__format_restaurant_info(restaurant_choice),
        )
        print_verbose(self.dialog_config["verbose"], "---- Other options ----")

        if other_options is None:
            print_verbose(self.dialog_config["verbose"], "None")
            return restaurant_choice, other_options

        for i, option in enumerate(other_options):
            print_verbose(self.dialog_config["verbose"], f"Option {i + 1}:")
            print_verbose(
                self.dialog_config["verbose"],
                self.__format_restaurant_info(option),
            )

        return restaurant_choice, other_options

    def __additional_preferences(self, candidate_restaurants, requirements):
        """Function to filter by additional preferences

        Arguments:
            Candidate restaurants: pd.dataframe
            Requirements: dictionary with keys
                "touristic": boolean
                "assigned_seats": boolean
                "children": boolean
                "romantic": boolean

        Requirements are satisfied or not based on values in candidate_restaurants dataframe
        Below are the variable names and values used for this code
            "pricerange": "cheap"
            "food_quality": "good"
            "food": "romanian"
            "crowdedness": "busy"
            "length_of_stay": "long"

        Outputs are
            restaurant_choice: one-line pd.dataframe with the chosen restaurant
            other_options: pd.dataframe with other options in case there are any
            reasons_all: list of dictionaries, reasons for choice being made; can be queried
                by looking for the specific restaurant name as a value in the "restaurantname" key
                for all dictionaries in the list

        Raises:
            LookupError: If no restaurant is found

        """

        chosen_restaurants = pd.DataFrame()  # final list of restaurants
        reasons_all = []  # collection of reasoning for all final choices
        restaurant_choice = None  # if multiple options, choose one
        other_options = None  # save other options in this variable

        for (
            restaurant
        ) in candidate_restaurants:  # loop over preselected restaurant options
            reasons = {}  # reasons for specific restaurant
            print_verbose(self.dialog_config["verbose"], requirements)
            if requirements["touristic"]:
                if (
                    restaurant["pricerange"] == "cheap"
                    and restaurant["food_quality"] == "good"
                ):
                    reasons["touristic"] = "It has cheap and good food."
                    print_verbose(self.dialog_config["verbose"], reasons)
                if restaurant["food"] == "romanian":
                    continue

            # Check if preference is explicitly stated
            elif requirements["touristic"] is False:
                if restaurant["food"] == "romanian":
                    reasons[
                        "not touristic"
                    ] = "It serves Romanian food, which tourists don't enjoy."  # add this to reasoning
                elif (
                    restaurant["pricerange"] == "cheap"
                    and restaurant["food_quality"] == "good"
                ):
                    continue  # restaurant touristic, try next one

            if requirements["assigned_seats"]:
                if restaurant["crowdedness"] == "busy":
                    reasons["assigned seats"] = "busy"
                else:
                    continue

            elif requirements["assigned_seats"] is False:
                if restaurant["crowdedness"] == "busy":
                    continue  # if you don't want assigned seats, busy restaurant will not work
                else:
                    reasons["no assigned seats"] = "The restaurant is not busy."

            if requirements["children"]:
                if restaurant["length_of_stay"] == "long":
                    continue  # if long stay, then no children --> check next restaurant
                else:
                    reasons["children"] = "The restaurant is meant for short stays."

            elif requirements["children"] is False:
                if restaurant["length_of_stay"] == "long":
                    reasons["no children"] = "The restaurant is meant for long stays."
                else:
                    continue

            if requirements["romantic"]:
                if (
                    restaurant["crowdedness"] != "busy"
                    and restaurant["length_of_stay"] == "long"
                ):
                    reasons[
                        "romantic"
                    ] = "The restaurant is not busy and allows for a long stay."
                elif restaurant["length_of_stay"] != "long":
                    continue
                elif restaurant["crowdedness"] == "busy":
                    continue

            elif requirements["romantic"] is False:
                if (
                    restaurant["crowdedness"] != "busy"
                    and restaurant["length_of_stay"] == "long"
                ):
                    continue
                elif (
                    restaurant["crowdedness"] == "busy"
                    and restaurant["length_of_stay"] != "long"
                ):
                    reasons[
                        "not romantic"
                    ] = "The restaurant is busy and only allows short stays."
                elif restaurant["crowdedness"] == "busy":
                    reasons["not romantic"] = "The restaurant is busy."
                elif restaurant["length_of_stay"] != "long":
                    reasons[
                        "not romantic"
                    ] = "The restaurant only allows for short to medium stays."
                print_verbose(self.dialog_config["verbose"], reasons)
            chosen_restaurants = pd.concat(
                [chosen_restaurants, pd.DataFrame([restaurant])]
            )  # add restaurant to final list

            reasons["restaurantname"] = restaurant[
                "restaurantname"
            ]  # mark reasons for specific restaurant
            reasons_all.append(reasons)
        if len(chosen_restaurants) == 0:
            return None, None, None
        elif len(chosen_restaurants) == 1:
            restaurant_choice = chosen_restaurants.iloc[0].to_dict()
        elif len(chosen_restaurants) > 1:
            restaurant_choice = chosen_restaurants.sample(n=1).iloc[0].to_dict()
            restaurant_choice_name = restaurant_choice["restaurantname"]
            other_options = chosen_restaurants[
                chosen_restaurants["restaurantname"] != restaurant_choice_name
            ].to_dict("records")

        return restaurant_choice, other_options, reasons_all

    def __format_restaurant_info(self, restaurant):
        """Function which formats the restaurant information

        Args:
            restaurant (_type_): The retrieved restaurant
        """
        if restaurant is not None:
            return dedent(
                f"""\
                Name: {restaurant['restaurantname']}
                Address: {restaurant['addr']}
                Postcode: {restaurant['postcode']}
                Phone number: {restaurant['phone']}
                Type of food: {restaurant['food']}
                Price range: {restaurant['pricerange']}
                Area: {restaurant['area']}"""
            )
        return "I'm sorry, I don't know any restaurants that match your preferences."

    def __prompt_other_preferences(self):
        """Function which prompts the user for other preferences"""
        # Check for which preferences we have to prompt
        if self.stored_preferences["food"] is None:
            self.__respond("What type of food do you prefer?")
        elif self.stored_preferences["pricerange"] is None:
            self.__respond("What price range do you prefer?")
        elif self.stored_preferences["area"] is None:
            self.__respond("What area of town do you prefer?")

    def __prompt_additional_preferences(self):
        """Function which prompts the user for additional preferences"""
        self.__respond(
            "Do you have any additional preferences? I can help you find a restaurant that is:"
            + "\n- (Not) touristic"
            + "\n- Has (no) assigned seats"
            + "\n- Is (not) child friendly"
            + "\n- Is (not) romantic"
        )

    def __get_suggestion_string(self, restaurant):
        """Function which returns a string with a restaurant suggestion

        Args:
            restaurant (_type_): The retrieved restaurant
        """
        suggestion_prefixes = get_suggestion_prefixes()
        if restaurant is not None:
            # Remove new lines from the returned string
            return dedent(
                f"""\
                {suggestion_prefixes[random.randint(0, len(suggestion_prefixes) - 1)]}
                {capitalize_first_letter(restaurant['restaurantname'])}.
                It's {self.__get_word_prefix(restaurant['pricerange'])}
                {'moderately priced' if restaurant['pricerange'] == 'moderate' else restaurant['pricerange']}
                restaurant and serves {capitalize_first_letter(restaurant['food'])} food.
                It is located in the {restaurant['area']} of town."""
            ).replace("\n", " ")
        return "I'm sorry, I don't know any restaurants that match your preferences."

    # -------------- Word helper methods --------------
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
            # If distance is less than a certain distance (default = 2), then we have a match
            if dist <= self.dialog_config["levenshtein"]:
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
                self.__add_message(None, match["option"], "Bot")

    def __get_word_prefix(self, word):
        """Function which a or an based on the first letter of a word

        Args:
            word (str): The word to get the prefix of
        """
        return "an" if word[0] in ["a", "e", "i", "o", "u"] else "a"

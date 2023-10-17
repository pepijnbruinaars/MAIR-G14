import argparse
from dialog_manager import DialogManager, DialogConfig
from helpers import check_models


def main(args: argparse.Namespace):
    # Check validity of model
    check_models(args)

    # Create dialog manager
    dialog_config: DialogConfig = vars(args)
    dialog_manager = DialogManager(dialog_config)

    # Start dialog
    dialog_manager.start_dialog()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant recommendation chatbot")
    parser.add_argument(
        "-m",
        "--model",
        help="Select the classification model to be used: RF (Random Forest), MLP (multilayer perceptron), Majority",
        default="neural",
        dest="intent_model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print out debug information",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--tts",
        help="Play system messages as speech",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--caps",
        help="Convert system output to all caps",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--levenshtein",
        help="Define the levenshtein distance",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-d",
        "--delay",
        help="Configure a delay before system response",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--speech",
        help="Takes the user input in the form of speech instead of text",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--typing-speed",
        help="Configure the typing speed multiplier of the system. Larger number means faster typing.",
        type=float,
        default=2,
    )
    parser.add_argument(
        "-g", "--gender",
        help="Gender' of the bot",
        type=str,
        default="none"
    )
    args = parser.parse_args()
    main(args)

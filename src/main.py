import argparse
from dialog_manager import DialogManager, DialogConfig
from helpers import check_models


def main(args: argparse.Namespace):
    # Check validity of model
    check_models()

    # Create dialog manager
    dialog_config: DialogConfig = {"intent_model": args.model, "verbose": args.verbose}
    dialog_manager = DialogManager(dialog_config)

    # Start dialog
    dialog_manager.start_dialog()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant recommendation chatbot")
    parser.add_argument(
        "-m",
        "--model",
        help="Select the classification model to be used",
        default="RF",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print out debug information",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)

import argparse
import os
from dialog_manager import DialogManager, DialogConfig


def main(args: argparse.Namespace):
    allowed_models = [
        # TODO: Uncomment when implemented
        # "keyword",
        "RandomForest",
        # 'neural'
    ]

    # Verify model
    if args.model not in allowed_models:
        print(f"Invalid model: {args.model}")
        return

    # Check models folder for first time use
    if args.model == "RF":
        with os.scandir("models") as folder:
            # If folder contains optimized_random_forest.joblib, then we are good to go
            if "optimized_random_forest.joblib" in [file.name for file in folder]:
                pass
            # Train model if not
            else:
                raise Exception(
                    "Intent classification model not found. Please run the random_forest.py script to train the model."
                )

    dialog_config: DialogConfig = {"intent_model": args.model, "verbose": args.verbose}

    dialog_manager = DialogManager(dialog_config)

    dialog_manager.start_dialog()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant recommendation chatbot")
    parser.add_argument(
        "-m",
        "--model",
        help="Select the classification model to be used",
        default="RandomForest",
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

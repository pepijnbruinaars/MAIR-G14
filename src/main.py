import os
from dialog_manager import DialogManager, DialogConfig


def main():
    # Check models folder for first time use
    with os.scandir("models") as folder:
        # If folder contains optimized_random_forest.joblib, then we are good to go
        if "optimized_random_forest.joblib" in [file.name for file in folder]:
            pass
        # Train model if not
        else:
            raise Exception(
                "Intent classification model not found. Please run the random_forest.py script to train the model."
            )

    dialog_config: DialogConfig = {"intent_model": "RandomForest", "verbose": True}

    dialog_manager = DialogManager(dialog_config)

    dialog_manager.start_dialog()


if __name__ == "__main__":
    main()

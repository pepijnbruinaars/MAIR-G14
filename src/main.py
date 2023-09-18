from dialog_manager import DialogManager


def main():
    print("Hello World!")
    
    dialog_config = {
        "intent_model": "RandomForest",
        "verbose": True
    }
    
    dialog_manager = DialogManager(dialog_config)
    
    dialog_manager.start_dialog()
    
if __name__ == "__main__":
    main()
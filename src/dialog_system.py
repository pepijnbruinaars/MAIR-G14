import csv
import argparse

from keyword_matching.keyword_matching import match_sentence
from testing.test_model import test_models

# Allowed intent classification methods
allowed_models = [
    # 'majority',
    'keyword',
    # 'tree',
    # 'neural'
]

def main(args):
    if args.test:
        # Run test suite
        test_models()
        return
    
    # Verify model
    if args.model not in allowed_models:
        print(f"Invalid model: {args.model}")
        return
    
    selected_model = args.model
    
    
    # Main loop
    running = True
    print(f"The selected model is {selected_model}")
    print("Hello, how can I help?")
    while running:
        user_input = input().lower()
        
        if user_input == "exit":
            running = False
            continue
        
        response = "" # Classifier classification
        
        # Switch through models
        match selected_model:
            case 'majority':
                response = "majority"
            case 'keyword':
                response = match_sentence(user_input)
            case 'tree':
                response = "tree"
            case 'neural':
                response = "neural"
            case _:
                response = "error"


        print(f"Predicted intent: {response}")
        print("Can I help you with anything else?")

if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(description='Restaurant recommendation chatbot')
    parser.add_argument('-m', '--model', help="Select the classification model to be used")
    parser.add_argument('-t', '--test', help="Run the test suite", action='store_true')
    args = parser.parse_args()
    main(args)
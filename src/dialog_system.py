import csv
import argparse

from keyword_matching.keyword_matching import match_sentence

# Allowed intent classification methods
allowed_models = [
    # 'majority',
    'keyword',
    # 'tree',
    # 'neural'
]

def load_csv_data(filepath):
    data = {}
    with open(filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None) # skip header
        
        # Load csv data into dict sorted by label
        for row in csv_reader:
            label = row[0]
            text = row[1]
            if label in data.keys():
                data[label].append(text)
            else:
                data[label] = [text]
        
    return data

def main(args):
    # Verify model
    if args.model not in allowed_models:
        print(f"Invalid model: {args.model}")
        return
    
    selected_model = args.model
    
    # Main loop
    running = True
    print(f"The selected model is {selected_model}")
    print("Hello how can I help?")
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
    args = parser.parse_args()
    main(args)
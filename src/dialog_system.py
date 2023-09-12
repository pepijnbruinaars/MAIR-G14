import csv
import argparse

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
    # Main loop
    running = True
    print(f"Hello how can I help?")
    while running:
        usr_input = input().lower()
        
        if usr_input == "exit":
            running = False
            continue
        
        response = "" # Classifier classification

        print("Can I help you with anything else?")

if __name__ == '__main__':
    # Set up argparser
    parser = argparse.ArgumentParser(description='Restaurant recommendation chatbot')
    parser.add_argument('-m', '--model', help="Select the classification model to be used")
    args = parser.parse_args()
    main(args)
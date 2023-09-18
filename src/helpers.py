import string
from nltk.corpus import stopwords

import csv

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

def save_data_as_csv(filepath, data_dict):
    with open(filepath, 'w') as f:
     
        write = csv.writer(f)
        
        write.writerow(["label", "text"])
        for key in data_dict.keys():
            for entry in data_dict[key]:
                row = [key]
                row.append(entry)
                write.writerow(row)

def remove_stopwords(data_dict):
    stop_words = set(stopwords.words('english'))

    for key, entries in data_dict.items():
        data_dict[key] = [" ".join([word for word in entry.split() if word not in stop_words])\
                           for entry in entries]

    return data_dict

def prep_user_input(user_input: str):
    # Remove stopwords
    user_input = " ".join([word for word in user_input.split() if word not in stopwords.words('english')])
    
    # Remove punctuation
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    user_input = user_input.lower()
    
    return user_input
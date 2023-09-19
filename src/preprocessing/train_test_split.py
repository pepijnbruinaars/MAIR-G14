import csv

from sklearn.model_selection import train_test_split

with open("data/dialog_acts.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader, None)  # skip header

    # Load csv data into dict sorted by label
    data = {}
    for row in csv_reader:
        label = row[0]
        text = row[1]
        if label in data.keys():
            data[label].append(text)
        else:
            data[label] = [text]

    # Split data into train and test sets
    train_data = {}
    test_data = {}
    for label in data.keys():
        train_data[label], test_data[label] = train_test_split(
            data[label], test_size=0.15
        )

    # Write train data to file
    with open("data/train_dialog_acts.csv", "w") as train_file:
        train_writer = csv.writer(train_file, delimiter=",")
        train_writer.writerow(["label", "text"])
        for label in train_data.keys():
            for text in train_data[label]:
                train_writer.writerow([label, text])

    # Write test data to file
    with open("data/test_dialog_acts.csv", "w") as test_file:
        test_writer = csv.writer(test_file, delimiter=",")
        test_writer.writerow(["label", "text"])
        for label in test_data.keys():
            for text in test_data[label]:
                test_writer.writerow([label, text])

    print("Done")

with open("data/no_duplicates_dialog_acts.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader, None)  # skip header

    # Load csv data into dict sorted by label
    data = {}
    for row in csv_reader:
        label = row[0]
        text = row[1]
        if label in data.keys():
            data[label].append(text)
        else:
            data[label] = [text]

    # Split data into train and test sets
    train_data = {}
    test_data = {}
    for label in data.keys():
        print(label)
        print(len(data[label]))
        try:
            train_data[label], test_data[label] = train_test_split(
                data[label], test_size=0.15
            )
        except ValueError:
            print(f"Label {label} has only one sample")
            continue

    # Write train data to file
    with open("data/train_dialog_acts_no_dupes.csv", "w") as train_file:
        train_writer = csv.writer(train_file, delimiter=",")
        train_writer.writerow(["label", "text"])
        for label in train_data.keys():
            for text in train_data[label]:
                train_writer.writerow([label, text])

    # Write test data to file
    with open("data/test_dialog_acts_no_dupes.csv", "w") as test_file:
        test_writer = csv.writer(test_file, delimiter=",")
        test_writer.writerow(["label", "text"])
        for label in test_data.keys():
            for text in test_data[label]:
                test_writer.writerow([label, text])

    print("Done")

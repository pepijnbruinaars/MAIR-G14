import csv

str = open(r"data/dialog_acts.dat", 'r')
str = str.read()

lines = str.splitlines()

processed_data = {}
for line in lines:
    to_split = line.find(' ') # Finds the first space in the line
    label = line[:to_split]
    text = line[to_split+1:]

    if label in list(processed_data.keys()):
        # Add to existing set (set does not allow duplicated)
        processed_data[label].add(text)
    else:
        # Add label to dict
        processed_data[label] = set([text])

with open('data/no_duplicates_dialog_acts.csv', 'w') as f:
     
    write = csv.writer(f)
     
    write.writerow(["label", "text"])
    for key in processed_data.keys():
        for entry in processed_data[key]:
            row = [key]
            row.append(entry)
            write.writerow(row)


    
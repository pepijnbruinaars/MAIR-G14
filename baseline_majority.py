import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/dialog_acts.csv")

majority_label = df["label"].value_counts().idxmax() # what is the majority label?
value_counts = df["label"].value_counts() # all label counts
majority_label_count = value_counts[majority_label] #majority label count

total_entries = len(df) # number of total entries

majority_label_ratio = majority_label_count/total_entries # ratio of majority label to all entries

print("Total number of entries:", total_entries)
print("Majority label:", majority_label, ", Number of occurences:", majority_label_count)
print("Ratio majority label/all data points:", round(majority_label_ratio, 2))

# Get X and y
X = df["text"]
y = df["label"]

# split data into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# baseline model always predicting the same outcome
def prediction_func(X):

    return majority_label


y_pred = X_test.apply(prediction_func)
y_true = y_test

# couldn't get the sklearn accuracy function to work so i made my own
true_or_false = []
# get accuracy score
for prediction, actual in zip(y_pred, y_true):
    if prediction == actual:
        true_or_false.append(1)
    else:
        true_or_false.append(0)

accuracy_score = true_or_false.count(1)/len(true_or_false)
print("Accuracy_score:", round(accuracy_score, 2))
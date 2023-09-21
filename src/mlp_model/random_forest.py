# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:32:40 2023

@author: 13vic
"""
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# import numpy as np
from sklearn.ensemble import RandomForestClassifier

# from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    # confusion_matrix,
    # precision_score,
    # recall_score,
    # ConfusionMatrixDisplay,
)


# Import training data without duplicates
train_data_no_dupes = pd.read_csv("data/splits/train_dialog_acts_no_dupes.csv")

# Import training data with duplicates
train_data = pd.read_csv("data/splits/train_dialog_acts.csv")


def process_data(df):
    # categorize the label data as numerical data, (null = -1), using pd.factorize
    df["label"] = pd.factorize(df["label"])[0]

    # Use the Sklearn method of countVectorizer to make a matrix of word counts
    # this method also tokenizes implicitly
    vectorizer = CountVectorizer()
    bag_of_words_matrix = vectorizer.fit_transform(df["text"])

    # From the matrix we can build the bag of words representation
    # we use the words as column names
    bag_of_words = bag_of_words_matrix.toarray()
    features = vectorizer.get_feature_names_out()
    training_data = pd.DataFrame(data=bag_of_words, columns=features)
    # Save training data to csv
    training_data.to_csv("data/splits/rf_training_data.csv")
    # Organize the data into the featuress (X) and target (y)
    x = training_data
    y = df["label"]

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )

    return x_train, x_test, y_train, y_test


def fit_random_forest(x, y):
    # X being the features and y the target

    # fit the model to the data
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x, y)

    return rf


def optimize_hyperparameters(x, y, searching=True):
    # Number of trees
    n_estimators = [i for i in range(300, 1000, 100)]
    # Number of features to consider at every split
    max_features = ["sqrt"]
    # Maximum depth of trees
    max_depth = [i for i in range(40, 100, 20)]
    max_depth.append(None)
    # Minimum number of samples to split a node
    min_samples_split = [4, 8]
    # Minimum number of samples at each leaf node
    min_samples_leaf = [1]
    # Method of selecting samples for training each tree
    bootstrap = [False]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    rf = RandomForestClassifier(random_state=42)

    # We randomly search parameter space for optimal values.
    # With 3 cross-validation and 100 attempts
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    # fit the model to the given training data if searching is set to true
    # if not set to searching we use the previously found best parameters
    if searching:
        rf_random.fit(x, y)
    else:
        # From function Optimizing_Hyperparamters we found the best parameters
        # by closing in on the best parameters over a series of optimizations
        rf = RandomForestClassifier(
            random_state=42,
            n_estimators=700,
            min_samples_split=8,
            min_samples_leaf=1,
            max_features="sqrt",
            max_depth=None,
            bootstrap=False,
        )
        rf.fit(x, y)
        return rf

    print("Best parameters found:\n", rf_random.best_params_)

    return rf_random.best_estimator_


best_parameters = {
    "n_estimators": 700,
    "min_samples_split": 8,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": None,
    "bootstrap": False,
}


def test_accuracy(DATA, FOREST=False, OPTIMIZED_FOREST=False):
    # Get the training data
    x_train, x_test, y_train, y_test = process_data(DATA)

    # Train the model, currently this is only the (optimized) forest model
    if FOREST:
        # Random Forest model trained on the data
        model = fit_random_forest(x_train, y_train)
        # Save model to models folder
        joblib.dump(model, "models/random_forest.joblib")
        chosen_model = "RandomForestClassifier"
    elif OPTIMIZED_FOREST:
        model = optimize_hyperparameters(x_train, y_train, False)
        # Save model to models folder
        joblib.dump(model, "models/optimized_random_forest.joblib")
        chosen_model = "OptimizedRandomForestClassifier"
    else:
        print("No valid model selected")
        return

    # predict values
    y_pred = model.predict(x_test)

    # test the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {chosen_model}", accuracy)


if __name__ == "__main__":
    print("Without duplicates")
    test_accuracy(train_data_no_dupes, FOREST=1)

    print("\nWithout duplicates and optimized")
    test_accuracy(train_data_no_dupes, OPTIMIZED_FOREST=1)

    # print("\nWith duplicates")
    # test_accuracy(train_data, FOREST = 1)

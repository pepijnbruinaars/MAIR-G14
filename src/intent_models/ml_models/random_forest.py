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
    """
    df = the dataframe you want to prepare for training

    test_split is the proportion of the data to use for testing purposes.
    if set to 0 all the data will go into training data

    Returns train test split, and the vectorizer fitted on training data as:

    x_train, y_train, x_test, y_test, vectorizer

    """
    # copy the dataframe to avoid mutating the original
    df_copy = df.copy()
    # categorize the label data as numerical data, (null = -1), using pd.factorize
    df_copy["label"] = pd.factorize(df_copy["label"])[0]
    

    # Use the Sklearn method of countVectorizer to make a matrix of word counts
    # this method also tokenizes implicitly
    vectorizer = CountVectorizer()
    bag_of_words_matrix = vectorizer.fit_transform(df_copy["text"])

    # From the matrix we can build the bag of words representation
    # we use the words as column names
    bag_of_words = bag_of_words_matrix.toarray()

    # With the bag of words represenation build a dataframe with features as
    # colomns
    features = vectorizer.get_feature_names_out()
    training_data = pd.DataFrame(data=bag_of_words, columns=features)

    # Save training data to csv
    training_data.to_csv("data/splits/rf_training_data.csv")

    # Organize the data into the featuress (X) and target (y)
    x = training_data
    y = df_copy["label"]

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=42
    )

    return x_train, x_test, y_train, y_test, vectorizer


def predict_single_input_rf(input):
    """
    Predict the intent of a single input string using the model.
    We have to check if the input contains words that are not in the training data, and
    ignore just those words if that's the case.
    Afterwards, we can use the model to predict the intent of the input, and we return
    the label of the intent.

    Args:
        input (__str__): The input string to predict the intent of.
    """
    # Get the training data
    labels = pd.factorize(train_data_no_dupes["label"])[1].tolist()
    labels.append("null")
    
    # load the vectorizer from data
    _, _, _, _, vectorizer = process_data(train_data_no_dupes)

    # Load the model
    model = joblib.load("models/optimized_random_forest.joblib")

    # Add the input to the vectorized training data
    input_data = vectorizer.transform([input]).toarray()

    features = vectorizer.get_feature_names_out()
    x_to_predict = pd.DataFrame(data=input_data, columns=features)

    # Predict the intent of the input
    y_pred = model.predict(x_to_predict)

    # Return the label of the intent
    return labels[y_pred[0]]


def fit_random_forest(x, y):
    # X being the features and y the target

    # fit the model to the data
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x, y)

    return rf


def optimize_hyperparameters(x, y, searching=True):
    """
    Parameters
    ----------
    x : dataframe
        the features the random_forest algorithm will train on.
    y : dataframe
        the target classification the random_forest will try to predict.
    searching : bool
        Make true if you want to search a predefined hyperparameter space for
        better options
        Make false if you want to use the best hyperparamters found so far

    Returns
    -------
    Random_forest_classifier model


    """
    if searching:
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

        rf_random.fit(x, y)

        print("Best parameters found:\n", rf_random.best_params_)

        return rf_random.best_estimator_

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


def generate_random_forest():
    """Generate the random forest model with optimized parameters. This is used solely for
    generating the model for the first time a user runs the system, and should not be used again.
    """
    # Get the training data
    x_train, _, y_train, _, _ = process_data(train_data_no_dupes)

    # Random forest model with optimized parameters, boolean for finding new parameters.
    model = optimize_hyperparameters(x_train, y_train, False)

    # Save model to models folder
    joblib.dump(model, "models/optimized_random_forest.joblib")

    return model


def test_accuracy(DATA, FOREST=False, OPTIMIZED_FOREST=False):
    # Get the training data
    x_train, x_test, y_train, y_test, _ = process_data(DATA)

    # Train the model, currently this is only the (optimized) forest model
    if FOREST:
        # Random Forest model trained on the data
        model = fit_random_forest(x_train, y_train)
        # Save model to models folder
        joblib.dump(model, "models/random_forest.joblib")
        chosen_model = "RandomForestClassifier"
    elif OPTIMIZED_FOREST:
        # Random forest model with optimized parameters, boolean for finding new parameters.
        model = generate_random_forest(DATA)
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

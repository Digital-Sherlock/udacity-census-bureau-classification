"""
Module that trains and saves a model.

Authors: Udacity | Vadim Polovnikov
Date: 2024-03-25
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import subprocess
import pickle
from ml import data, model


# Loading the data
try:
    dataset = pd.read_csv("../cleaned_data/census_cleaned.csv")
except FileNotFoundError:
    subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
    dataset = pd.read_csv("../cleaned_data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead
# of a train-test split.
train, test = train_test_split(dataset, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Tranforming training data
X_train, y_train, encoder, lb = data.process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Processing the test data
X_test, y_test, _, _ = data.process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = model.train_model(X_train, y_train)
pickle.dump(model, open("../model/model.pkl", "wb"))

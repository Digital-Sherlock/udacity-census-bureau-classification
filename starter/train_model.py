"""
Module that trains and saves a model.

Authors: Udacity | Vadim Polovnikov
Date: 2024-03-25
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import subprocess
from ml import data, model


# Loading the data
try:
    dataset = pd.read_csv("../cleaned_data/census_cleaned.csv")
except FileNotFoundError:
    subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
    dataset = pd.read_csv("../cleaned_data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(dataset, test_size=0.20, stratify=["salary"])

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
X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.

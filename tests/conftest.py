"""
Configuration file for Pytest to test against
model.py functions.

Author: Vadim Polovnikov
Date: 2024-03-30
"""

import sys
import os
import pytest
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split
import pickle
# Adding ml package to Python path
# Below will work if pytest executes from root project folder
sys.path.append(os.getcwd()); from ml.data import process_data
from constants import cat_features


@pytest.fixture(scope="function")
def data():
    """
    Retrieve census_cleaned.csv.

    Input:
        - None
    Output:
        - df: (pd.DataFrame) cleaned_data DF
    """
    try:
        df = pd.read_csv("./cleaned_data/census_cleaned.csv")
    except FileNotFoundError:
        subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
        df = pd.read_csv("./cleaned_data/census_cleaned.csv")

    return df


@pytest.fixture(scope="function")
def x_train_y_train():
    '''
    Fixture for train_model function.

    Input:
        - None
    Output:
        X_train: (numpy.ndarray) input vars for training
        y_train: (numpy.ndarray) target vars
    '''
    try:
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")
    except FileNotFoundError:
        subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")

    train, _ = train_test_split(dataset, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True)

    return X_train, y_train, encoder, lb


@pytest.fixture(scope="function")
def test_set():
    """
    Fixture to retrieve y_test for testing
    compute_model_metrics function.

    Input:
        - None
    Output:
        y: (numpy.ndarray) y_test array
    """
    # Pulling the cleaned dataset
    try:
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")
    except FileNotFoundError:
        subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")

    # Spliting the dataset
    train, test = train_test_split(dataset, test_size=0.20, random_state=42)

    # Getting encoder and lb for the test dataset encoding
    _, _, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True)

    # Extracting y_test
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb)

    return X_test, y_test


@pytest.fixture(scope="function")
def y_pred():
    """
    Fixture to make model predictions (y_pred) to
    use for compute_model_metrics testing.

    Input:
        - None
    Output:
        y_pred: (numpy.ndarray) model predictions
    """

    # Loading a model
    try:
        model = pickle.load(open("./model/model.pkl", "rb"))
    except FileNotFoundError:
        print("model.pkl doens't exist")

    # Pulling the cleaned dataset
    try:
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")
    except FileNotFoundError:
        subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
        dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")

    # Spliting the dataset
    train, test = train_test_split(dataset, test_size=0.20, random_state=42)

    # Getting encoder and lb for the test dataset encoding
    _, _, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label='salary',
        training=True)

    # Extracting y_test
    X_test, _, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb)

    # Making preditcions
    y_pred = model.predict(X_test)

    return y_pred


@pytest.fixture(scope="function")
def get_model():
    """
    Fixture for retreiving model.

    Input:
        - None
    Output:
        - model: (sklearn.linear_model._logistic.LogisticRegression)
                trained model
    """

    try:
        model = pickle.load(open("./model/model.pkl", "rb"))
    except FileNotFoundError:
        print("model.pkl doens't exist")

    return model

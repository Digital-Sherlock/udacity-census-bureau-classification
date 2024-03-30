"""
Pytest testing suite dedicated to model.py
functions testing.

Author: Vadim Polovnikov
Date: 2024-03-30
"""

#imports
from starter.ml.model import *
import numpy as np


def test_train_model(x_train_y_train, train_model):
    """
    Tests train_model function.

    Input:
        train_model: (function) tested function
        x_train_y_train: (function) output from Pytest fixture
    Output:
        model: (sklearn.linear_model._logistic.LogisticRegression)
                trained model
    """
    X_train, y_train, _, _ = x_train_y_train

    # Checking for input data
    assert isinstance(X_train, numpy.ndarray), "X_train has to be Numpy array"
    assert isinstance(y_train, numpy.ndarray), "y_train has to be Numpy array"
    assert X_train.shape[0] == y_train.shape[0], \
        "Number of input and target variables don't match"
    
    # Checking for expected data shapes
    assert X_train.shape[1] == 12, "Unexpected number of columns in X_train"
    assert y_train.shape[0] == 1, f"Unexpected shape of y_train - {y_train.shape}"
    
    model = train_model(X_train, y_train)

    return model


def test_compute_model_metrics(test_set, y_pred):
    """
    Tests compute_model_metrics function.

    Input:
        - test_set: (function) output from Pytest fixture
        - y_pred: (function) output from Pytest fixture
    Output:
        - 
    """
    # Retrieving data from fixtures
    _, y = test_set
    y_pred = y_pred

    # Testing input data types
    assert isinstance(y, numpy.ndarray), "Wrong datatype for y variable"
    assert isinstance(y_pred, numpy.ndarray), "Wrong datatype for y_pred"

    # Testing input shapes
    

    
"""
Pytest testing suite dedicated to model.py
functions testing.

Author: Vadim Polovnikov
Date: 2024-03-30
"""

import sys
import os
import numpy
# Adding ml package to Python path
# Below will work if pytest executes from root project folder
sys.path.append(os.getcwd()); from ml.model import *


def test_train_model(x_train_y_train):
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

    model = train_model(X_train, y_train)

    return model


def test_compute_model_metrics(test_set, y_pred):
    """
    Tests compute_model_metrics function.

    Input:
        - test_set: (function) output from Pytest fixture
        - y_pred: (function) output from Pytest fixture
    Output:
        - precision, recall, fbeta: (float) model scores
    """
    # Retrieving data from fixtures
    _, y = test_set

    # Testing input data types
    assert isinstance(y, numpy.ndarray), "Wrong datatype for y variable"
    assert isinstance(y_pred, numpy.ndarray), "Wrong datatype for y_pred"

    # Testing input shapes
    assert y.shape[0] == y_pred.shape[0], "y and y_pred have different shapes"

    # Getting model metrics
    precision, recall, fbeta = compute_model_metrics(y, y_pred)

    return precision, recall, fbeta


def test_inference(test_set, get_model):
    """
    Tests model.py inference function.

    Input:
        - test_set: (function) pytest fixture for
                    getting X_test
        - get_model: (function) pytest fixture for
                    getting model
        - inference: (function) tested function
    Output:
        - y_hat: (numpy.ndarray) predictions
    """

    # Loading test set
    X_test, _ = test_set

    # Checking data
    assert isinstance(X_test, numpy.ndarray)

    # Loading model from fixture
    model = get_model

    # Making predictions
    yhat = inference(model, X_test)

    if len(X_test) > 1:
        assert X_test.shape[0] == yhat.shape[0]
    else:
        # in case prediction is a single number
        assert isinstance(yhat, numpy.generic)

    return yhat

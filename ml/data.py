"""
This module is dedicated data preprocessing.

Authors: Udacity | Vadim Polovnikov
Date: 2024-03-25
"""


import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    lb=None,
    std_scaler=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical
    features and a label binarizer for the labels. This can be
    used in either training or inference/validation.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array
        will be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    std_scaler: sklearn.preprocessing._encoders.StandardScaler
        Trained sklearn StandardScaler, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns
        the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns
        the binarizer passed in.
    std_scaler: klearn.preprocessing._encoders.StandardScaler
        Trained StandardScaler if training is True, otherwise returns
        the scaler passed in.
    """

    # Separating input variables and label
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_numerical = X.drop(*[categorical_features], axis=1)

    # Encoding data during training
    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        std_scaler = StandardScaler()

        X_numerical = std_scaler.fit_transform(X_numerical)
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        X_numerical = std_scaler.transform(X_numerical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    # Putting numerical and encoded cat features together
    X = np.concatenate([X_numerical, X_categorical], axis=1)
    return X, y, encoder, lb, std_scaler

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import numpy


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Checking for input data
    assert isinstance(X_train, numpy.ndarray), "X_train has to be Numpy array"
    assert isinstance(y_train, numpy.ndarray), "y_train has to be Numpy array"
    assert X_train.shape[0] == y_train.shape[0], \
        "Number of input and target variables don't match"

    # Initializing a model
    lr = LogisticRegression(random_state=42)

    # Training
    lr.fit(X_train, y_train)

    return lr


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = f1_score(y, preds)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    yhat : np.array
        Predictions from the model.
    """

    yhat = model.predict(X)

    return yhat

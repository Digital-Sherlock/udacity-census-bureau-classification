"""
Used to test main.py API endpoints.

Author: Vadim Polovnikov
Date: 2024-04-06
"""

from fastapi.testclient import TestClient
import numpy as np
import sys
import os
# Adjusting PYTHONPATH
# Below will work if pytest executes from root project folder
sys.path.append(os.getcwd())
from main import app


# Defining a client for testing
client = TestClient(app)


def test_greeter():
    """
    Tests greeter function from main.py

    Input:
        - None
    Output:
        - None
    """
    response = client.get('/')

    assert response.status_code == 200
    assert response.json() == {
        "msg": "Hey there! Welcome! Looking for some predictions?"}


def test_columns(data):
    """
    Test number of columns in census_cleaned.csv
    which main:make_predictions relies on.

    Input:
        - data: (pd.DataFrame) cleaned_census.csv DF
    Output:
        - None
    """
    data_variables = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'hours-per-week',
        'native-country',
        'salary'
    ]
    assert len(data.columns) == 13, "Invalid number of variables"
    assert set(data.columns).issubset(set(data_variables)), \
        "Unknown variables in the cleaned_census.csv dataset"


def test_make_predictions(data):
    """
    Tests make_predictions function from main.
    """
    # Using a random example from cleaned_census.csv
    data.pop("salary")
    testset_json = data.iloc[np.random.randint(0, len(data)), :].to_json()

    response = client.post(
        url="/inference",
        content=testset_json
    )

    assert response.status_code == 200
    assert response.json() == {"model_prediction": "0"} or \
        {"model_prediction": "1"}


def test_make_predictions_unknown_input_variables(data):
    """
    Tests make_predictions() when unknown input variables
    are sent.
    """
    sample_data = data.iloc[np.random.randint(0, len(data)), :]

    # Tweaking random feaure
    random_feature = np.random.randint(len(sample_data.index))
    for feature in range(random_feature):
        if isinstance(sample_data[feature], str):
            sample_data[feature] = -1
            response = client.post(
                url='/inference',
                content=sample_data
            )
            assert response.status_code == 400
        else:
            sample_data[feature] = "dummy"
            response = client.post(
                url='/inference',
                content=sample_data
            )
            assert response.status_code == 400

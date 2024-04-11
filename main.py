"""
API to interact with a model. In partilcar,
this API achieves a welcome message and model
inference.

Author: Vadim Polovnikov
Date: 2024-04-05
"""
from constants import cat_features
import sys
import os
import pandas as pd
import subprocess
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Adding current folder to PYTHONPATH
sys.path.append(os.getcwd())
from ml.model import inference
from ml.data import process_data


# Initializing the app
app = FastAPI()

# Defining a data structure for POST req


class InferenceBody(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get('/')
def greeter() -> dict:
    """
    API endpoint that greets a user.

    Input:
        - None
    Output:
        - (dict) welcome message
    """
    return {"msg": "Hey there! Welcome! Looking for some predictions?"}


@app.post('/inference')
def make_predictions(infset: InferenceBody) -> dict:
    """
    API used for model inference.

    Input:
        - infset: (class) object of InferenceBody
    Output:
        - (dict) model predictions
    """
    # Dowloading original dataset
    try:
        df_origin = pd.read_csv("./cleaned_data/census_cleaned.csv")
    except FileNotFoundError:
        subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
        df_origin = pd.read_csv("./cleaned_data/census_cleaned.csv")

    # Removing target variables
    df_origin.pop("salary")
    # De-serializes POST req body to python dict with alias
    infset_dict = infset.model_dump(by_alias=True)

    # Creating a dict of expected data ranges/values
    expected_values_dict = {}
    for feature in df_origin.columns:
        if df_origin[feature].dtype == 'int64':
            expected_values_dict[feature] = \
                (min(df_origin[feature]), max(df_origin[feature]))
        else:  # if dtype == 'object'
            expected_values_dict[feature] = df_origin[feature].unique()

    # Checking for valid data ranges/values
    for attr in list(infset_dict.keys()):
        if isinstance(infset_dict[attr], int):
            minimun = expected_values_dict[attr][0]
            maximum = expected_values_dict[attr][1]
            if infset_dict[attr] < minimun or infset_dict[attr] > maximum:
                raise HTTPException(
                    status_code=400,
                    detail=f"Value out of range {attr} ({infset_dict[attr]})")
        elif isinstance(infset_dict[attr], str):
            if infset_dict[attr] not in expected_values_dict[attr]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for {attr} ({infset_dict[attr]})"
                )

    # Convert dict to df for preprocessing
    index = len([list(infset_dict.values())[0]])
    df_infset = pd.DataFrame(infset_dict, index=[index])

    # Loading encoder and model
    try:
        # Since both of the below files depend on a single module -
        # train_model.py, will include them in a single try-except
        encoder = pickle.load(open("./model/encoder.pkl", "rb"))
        model = pickle.load(open("./model/model.pkl", "rb"))
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load model/encoder - {err}"
        )

    # Processing inference set
    X, _, _, _ = process_data(
        df_infset,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None)

    # Making predictions
    y_pred = inference(model, X)

    return {"model_prediction": str(y_pred[-1])}

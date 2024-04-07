"""
API to interact with a model. In partilcar,
this API achieves a welcome message and model
inference.

Author: Vadim Polovnikov
Date: 2024-04-05
"""
from ml.model import inference
from ml.data import process_data
import sys
import os
import pandas as pd
import subprocess
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Adding current folder to PYTHONPATH
sys.path.append(os.getcwd())


# Initializing the app
app = FastAPI()

# Defining a data structure of POST req


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

    df_origin_features = df_origin.columns
    # Serializes POST req body to python dict with alias
    infset_dict = infset.model_dump(by_alias=True)
    infset_features = list(infset_dict.keys())

    # Checking if passed features exist in the dataset
    # model was trained on
    if set(infset_features).issubset(set(df_origin_features)) == False:
        raise HTTPException(
            status_code=400,
            detail=f"""Body contains unknown input variables.
                Request parameters - {infset_dict}""")

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

    # Cat features for futher preprocessing
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

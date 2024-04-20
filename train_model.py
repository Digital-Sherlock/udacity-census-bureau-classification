"""
Module that trains and saves a model.

Authors: Udacity | Vadim Polovnikov
Date: 2024-03-25
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import subprocess
import pickle
import json
from ml import data, model
from constants import cat_features


# Loading the data
try:
    dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")
except FileNotFoundError:
    subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
    dataset = pd.read_csv("./cleaned_data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead
# of a train-test split.
train, test = train_test_split(dataset, test_size=0.20, random_state=42)
train.to_csv("./data/train_df.csv", index=False)
test.to_csv("./data/test_df.csv", index=False)

# Tranforming training data
X_train, y_train, encoder, lb, std_scaler = data.process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Processing the test data
X_test, y_test, _, _, _ = data.process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
    std_scaler=std_scaler
)

# Training and saving a model.
inference_artifact = model.train_model(X_train, y_train)
pickle.dump(inference_artifact, open("./model/model.pkl", "wb"))

# Saving transformers
pickle.dump(encoder, open("./model/encoder.pkl", "wb"))
pickle.dump(lb, open("./model/lb.pkl", "wb"))
pickle.dump(std_scaler, open("./model/std_scaler.pkl", "wb"))

# Saving training/testing data
pd.DataFrame(X_train).to_csv("./data/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv("./data/y_train.csv", index=False)
pd.DataFrame(X_test).to_csv("./data/X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("./data/y_test.csv", index=False)

if __name__ == "__main__":
    # Making predictions
    preds = model.inference(inference_artifact, X_test)

    # Computing metrics
    precision, recall, f1_score = model.compute_model_metrics(y_test, preds)

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    with open("./model/model_perf.json", "w") as file:
        json.dump(results, file)

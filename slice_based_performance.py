"""
This module defines slicing functions
for least representative categorical features.
It also outputs model perfolmance on these
slices.

Author: Vadim Polovnikov
Date: 2024-04-15
"""

import pandas as pd
import subprocess
from snorkel.slicing import slicing_function, PandasSFApplier
from snorkel.analysis import Scorer
from snorkel.utils import preds_to_probs
import pickle
from ml.model import inference


try:
    df_test = pd.read_csv("./data/test_df.csv")
except FileNotFoundError:
    subprocess.run(["python", "train_model.py"])


# Looking for young people in the gov positions
@slicing_function()
def young_people(x):
    return x.age < 30


@slicing_function()
def gov_workers(x):
    return x.workclass == 'State-gov' or x.workclass == 'Local-gov'


young_gov_workers_sfs = [young_people, gov_workers]

# Intializing PandasSFApplier
applier = PandasSFApplier(young_gov_workers_sfs)

# Sliced DF
S_test = applier.apply(df_test)

# Defining a scoring method
scorer = Scorer(metrics=['f1'])

# Getting predictions
model = pickle.load(open("./model/model.pkl", "rb"))
X_test = pd.read_csv("./data/X_test.csv")
preds = inference(model, X_test)
probs = preds_to_probs(preds, 2)

# Loading target variables
y_test = pd.read_csv("./data/y_test.csv")

# Scoring
score = scorer.score_slices(
    S=S_test,
    golds=y_test,
    preds=preds,
    probs=probs,
    as_dataframe=True
)

# Saving score df
score.to_csv("./model/slice_perf.csv")

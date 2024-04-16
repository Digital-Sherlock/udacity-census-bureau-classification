"""
This module defines slicing functions
for least representative categorical features.
It also outputs model perfolmance on these
slices.

Author: Vadim Polovnikov
Date: 2024-04-15
"""

import pandas as pd
import numpy as np
import subprocess
from snorkel.slicing import slicing_function, slice_dataframe
from constants import cat_features


try:
    df = pd.read_csv("./cleaned_data/census_cleaned.csv")
except FileNotFoundError:
    subprocess.run(["dvc", "pull", "-R", "--remote", "s3remote"])
    df = pd.read_csv("./cleaned_data/census_cleaned.csv")


def least_repr_cats(df=df, cat_features=cat_features):
    """
    This function identifies least
    representative categories and outputs
    them in a dictionary in the feature:cat
    pairs.

    Input:
        - dataframe: (pd.DataFrame) input DF
    Output:
        - least_rep_dict: (dict) dictionary
    """
    least_rep_dict = {}

    # DF made of cat features only
    cat_df = df.select_dtypes(include=[object])
    cat_df.pop('salary')

    for feature in cat_df.columns:
        least_rep_dict[feature] = cat_df[feature].value_counts().index[-1]

    return least_rep_dict


least_repr_dict = least_repr_cats()


# Defining SFs
@slicing_function()
def workclass_least_repr(x):
    """
    SF for least representative category
    in workclass feature.
    """
    return x['workclass'] == least_repr_dict['workclass']


@slicing_function()
def education_least_repr(x):
    """
    SF for least representative category
    in education feature.
    """
    return x['education'] == least_repr_dict['education']


@slicing_function()
def marital_status_least_repr(x):
    """
    SF for least representative category
    in marital-status feature.
    """
    return x['marital-status'] == least_repr_dict['marital-status']


@slicing_function()
def occupation_least_repr(x):
    """
    SF for least representative category
    in occupation feature.
    """
    return x['occupation'] == least_repr_dict['occupation']


@slicing_function()
def relationship_least_repr(x):
    """
    SF for least representative category
    in relationship feature.
    """
    return x['relationship'] == least_repr_dict['relationship']


@slicing_function()
def race_least_repr(x):
    """
    SF for least representative category
    in race feature.
    """
    return x['race'] == least_repr_dict['race']


@slicing_function()
def sex_least_repr(x):
    """
    SF for least representative category
    in sex feature.
    """
    return x['sex'] == least_repr_dict['sex']


@slicing_function()
def native_country_least_repr(x):
    """
    SF for least representative category
    in native_country feature.
    """
    return x['native_country'] == least_repr_dict['native_country']


# Defining a list of SFs

# I doubt there will be any data point that
# will satisfy them all; combining some of
# them might be a good idea
sfs = [
    workclass_least_repr,
    education_least_repr,
    marital_status_least_repr,
    occupation_least_repr,
    relationship_least_repr,
    race_least_repr,
    sex_least_repr,
    native_country_least_repr
]

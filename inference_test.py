"""
Quick and easy code to test /inference API

Author: Vadim Polovnikov
DateDate: 2024-03-26
"""

import requests


api_url = input("Enter API endpoint address: ")
sample = {"age": 33,
          "workclass": "Self-emp-not-inc",
          "fnlgt": 140729,
          "education": "Bachelors",
          "education-num": 13,
          "marital-status": "Married-civ-spouse",
          "occupation": "Farming-fishing",
          "relationship": "Husband",
          "race": "White",
          "sex": "Male",
          "hours-per-week": 35,
          "native-country": "United-States"}


req = requests.post(url=api_url, json=sample)

print(req.text)

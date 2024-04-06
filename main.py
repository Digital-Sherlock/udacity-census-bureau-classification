"""
API to interact with a model. In partilcar, 
this API achieves a welcome message and model
inference.

Author: Vadim Polovnikov
Date: 2024-04-05
"""

from fastapi import FastAPI


# Initializing the app
app = FastAPI()

@app.get('/')
def greeter() -> str:
    """
    Greets a user.
    """
    return "Hey there! Welcome! Looking for some predictions?"


@app.post('/inference')
def inference():
    pass
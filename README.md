# udacity-census-bureau-classification

This project's objective is to incorporate CI/CD concepts into ML project lifecycle. The model's task is to predict whether a given person earns more than 50K of US dollars based on the Census Bureau dataset. Continuous Integration (CI) is built on GitHub Actions platform with Heroku-integrated Continuous Delivery (CD). Additional tools used throughout the project are AWS S3 for remote data storage, DVC for data versioning, FastAPI for inference, and Snorkel for slice-based learning. 

# Environment Set up
* Conda is used throughout the course of the project, however, one may choose virtual environment management tool of her liking. requirements.txt file is supplied. 

## Repositories

* Create a directory for the project and initialize Git and DVC.

## Set up S3

AWS S3 bucket is used for remote storage. However, other options are available for DVC. Refer to the official documentation.

For the ones who decide to stick with AWS S3 see instructions below:

* Create an S3 bucket under AWS S3 service section
* Sign in to the IAM console
* In the left navigation bar select Users, then choose Add user.
* Give the user a name and select Programmatic access.
* In the permissions selector, search for S3 and give it AmazonS3FullAccess
* After reviewing your choices, click create user.
* Configure your AWS CLI to use the Access key ID and Secret Access key.

# Project Use

Below section describes main components of the project.

## Model Training

Model training is performed via **train_model.py** module. It results in creation of the inference artifact itself as well as encoders - encoder.pkl, lb.pkl, model.pkl, std_scaler.pkl - under the model folder.

## Slice-based Learning

**slice_based_perfomance.py** is using Snorkel library for calculating model performance on slices of data. A selected slice for this project is young people working government positions. The result of model performance is stored under model folder as **slice_output.txt** which can be compared to overal score **model_perf.json**. Metrics used for evaluating the model were F1 score, precision, and recall.

## Model Inference

Model inference is done via **model.py** module of the **ml** package. One can retrieve the results by calling /inference API endpoint configured in **main.py** module using FastAPI. 

To start API use **uvicorn main:app** command.
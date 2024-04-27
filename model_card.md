# Model Card

## Model Details

### Overview

This is a Linear Regression mode dedicated for learning purposes only and not meant for production environments. It iss trained on the Census bureau classification dataset.

### Owners

Vadim Polovnikov (vpol@example.cpl)

## Intended Use

This model is inteded for studing machine learning principles and specifically concepts of how one can "fit it" the model into a production environemnt following MLOps methodologies.

## Training Data

Training data was received via Scikit-Learn train_test_split function with not stratification. Training data cab be found under data folder (train_df.csv). It's comprised of 80% of the original dataset.

## Evaluation Data

Evaluation data was received via Scikit-Learn train_test_split function with not stratification. Evaluation data cab be found under data folder (test_df.csv). It's comprised of 20% of the original dataset.

## Metrics

Model performance on the selected metrics:

Precision: 0.71328125

Recall: 0.569557080474111

f1_score: 0.6333680194242108

## Ethical Considerations

This model wasn't intended to produce the most accurate results but solely as a part of the production deployment process training, hence, its predictions should be considered in that way.

## Caveats and Recommendations

Shouldn't be considered real-world production model. One can use it to further improve its predictions on the given dataset for learning purposes.

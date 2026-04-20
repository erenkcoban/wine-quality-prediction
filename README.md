# Wine Quality Prediction App

This project is a Streamlit-based machine learning application that predicts red wine quality using multiple classification models. Users can upload a CSV dataset, choose preprocessing options, evaluate model performance, and make custom predictions through an interactive interface.

## Features

- Upload a CSV dataset
- Select a machine learning model:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Apply preprocessing techniques:
  - Noise injection
  - Random Over-Sampling
  - SMOTE
  - MinMaxScaler
  - StandardScaler
- View model performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
- Perform K-Fold cross validation
- Make custom predictions using sidebar input controls

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- imbalanced-learn

## Project Structure

```text
.
├── makine_proje.py
├── winequality-red.csv
└── README.md

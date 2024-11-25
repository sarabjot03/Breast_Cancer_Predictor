# Breast Cancer Prediction App

This repository contains a machine learning application to predict whether a breast cancer tumor is malignant or benign. The application is built using Python and Streamlit, utilizing a trained neural network model.

## Overview

The project involves:
- Loading and preprocessing the Breast Cancer dataset from scikit-learn.
- Performing feature selection to identify the most significant features.
- Training a Multi-layer Perceptron (MLP) classifier with hyperparameter tuning.
- Deploying the model in a Streamlit app for user interaction and predictions.

## Features

- **Data Preprocessing**: Normalization and feature selection using ANOVA F-test.
- **Model Training**: MLPClassifier with grid search for hyperparameter optimization.
- **User Interface**: Interactive Streamlit app for inputting feature values and receiving predictions.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   cd breast-cancer-prediction

Install Dependencies
Ensure you have Python 3.x installed. Then, install the required packages:
bash
pip install -r requirements.txt

Prepare the Environment
Ensure scaler.pkl and trained_model.pkl are present in the project directory. These files contain the pre-trained model and scaler.
Run the Application
Launch the Streamlit app using:
bash
streamlit run app.py

Usage
Open your web browser and navigate to http://localhost:8501.
Use the sliders in the sidebar to input feature values.
Click on "Predict" to see whether the tumor is predicted to be malignant or benign.
Files
app.py: The main script for running the Streamlit application.
Breast_cancer_assignment.ipynb: Jupyter notebook containing data processing, model training, and evaluation code.
scaler.pkl: Serialized scaler object used for data normalization.
trained_model.pkl: Serialized MLP model trained on selected features.
Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

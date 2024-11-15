import joblib
import sys
import pandas as pd
import numpy as np
from utils import load_object
from src.exception import CustomException
from src.logger import logging

def predict(input_df):
    """
    Predicts sentiment labels for a given input dataframe of reviews using a pre-trained model.

    Parameters:
    - input_df (pd.DataFrame): A DataFrame containing two columns: 'Review'，'Sentiment'.

    Returns:
    - binary_predictions (np.array): An array of binary sentiment predictions.
    """
    try:
        logging.info("Loading input data")
        X_test = np.array([np.fromstring(row, sep=' ') for row in input_df['Review']])
        y_test = input_df['Sentiment'].to_numpy()

        logging.info("Loading model")
        model = load_object(file_path = 'outputs/models/model.pkl')

        logging.info("Predicting results")
        y_pred             = model.predict(X_test)
        binary_predictions = (y_pred >= 0.5).astype(int)

        return binary_predictions
    
    except Exception as e:
            raise CustomException(e, sys)

def main():
    """
    Loads test data, makes predictions, and saves the results.

    Purpose:
    - Serves as an entry point to execute the prediction pipeline end-to-end, from loading test data to saving predictions.
    """
    test_pr = pd.read_csv('data/processed/test_processed.csv')
    preds   = predict(test_pr)

    pd.DataFrame(preds).to_csv('outputs/predictions/predictions.csv', index = False)

if __name__ == "__main__":
    main()


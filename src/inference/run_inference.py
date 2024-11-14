import joblib
import pandas as pd
from src.utils import load_object

def predict(X_test, y_test):
    model = load_object(file_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/outputs/models/model.pkl')

    y_pred             = model.predict(X_test)
    binary_predictions = (y_pred >= 0.5).astype(int)

    return binary_predictions

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import joblib
import pandas as pd
from data_loader import load_data
from data_processing import preprocess_text
from evaluation import evaluate_model
from feature_engineering import encode_labels
from utils import load_config
import logging

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        config = load_config()

        models_path      = config['outputs']['models']
        figures_path     = config['outputs']['figures']
        predictions_path = config['outputs']['predictions']

        # Create output directories
        os.makedirs(models_path, exist_ok = True)
        os.makedirs(figures_path, exist_ok = True)
        os.makedirs(predictions_path, exist_ok = True)
        logger.info("Output directories created")

        # Load data
        _, test_df = load_data()

        # Preprocess text (Stemming)
        logger.info("Preprocessing test data")
        test_df['processed_review'] = [preprocess_text(text) for text in test_df['review']]

        # Load vectorizer and model
        vectorizer_path = f"{models_path}/tfidf_vectorizer.pkl"
        model_path = f"{models_path}/lr_stemmed_tfidf.pkl"
        
        logger.info(f"Loading TF-IDF vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Vectorize test data
        logger.info("Vectorizing test data")
        X_test = vectorizer.transform(test_df['processed_review'])

        # Encode labels
        
        y_test, _ = encode_labels(test_df['sentiment'])

        # Perform inference
        logger.info("Performing inference")
        y_pred = model.predict(X_test)

        # Save predictions
        predictions_df = pd.DataFrame({
            'review'         : test_df['review'],
            'true_label'     : test_df['sentiment'],
            'predicted_label': y_pred
        })
        predictions_path = f"{predictions_path}/predictions.csv"
        predictions_df.to_csv(predictions_path, index = False)
        logger.info(f"Predictions saved to {predictions_path}")

        # Evaluate model
        logger.info("Evaluating model on test set")
        metrics = evaluate_model(model, X_test, y_test, "Tuned Logistic Regression (Stemmed + TF-IDF) (Test)", safe_results = True)

        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
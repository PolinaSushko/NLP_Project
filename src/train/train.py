import os
import sys
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, src_path)

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_loader import load_data
from data_processing import preprocess_text
from feature_engineering import vectorize_data, encode_labels
from evaluation import evaluate_model
from utils import load_config
import logging

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        config = load_config()

        models_path  = config['outputs']['models']
        figures_path = config['outputs']['figures']

        # Create output directories
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(figures_path, exist_ok=True)
        logger.info("Output directories created")

        # Load data
        train_df, test_df = load_data()
        
        # Split training data into train and validation sets
        logger.info("Splitting training data into train and validation sets")
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Preprocess text (Stemming)
        logger.info("Preprocessing training and validation data")
        train_df['processed_review'] = [preprocess_text(text) for text in train_df['review']]
        val_df['processed_review']   = [preprocess_text(text) for text in val_df['review']]

        # Encode labels
        y_train, label_encoder = encode_labels(train_df['sentiment'])
        y_val, _ = encode_labels(val_df['sentiment'])

        # Vectorize data
        X_train, _, vectorizer = vectorize_data(train_df['processed_review'])
        X_val = vectorizer.transform(val_df['processed_review'])

        # Save vectorizer
        vectorizer_path = f"{models_path}/tfidf_vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")

        # Define and train model with hyperparameter tuning
        logger.info("Starting model training with hyperparameter tuning")
        param_grid = [
            {'solver': ['liblinear'], 'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},
            {'solver': ['lbfgs'], 'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
        ]
        model = LogisticRegression(random_state = 42, max_iter = 1000)
        grid_search = GridSearchCV(model, param_grid, scoring = 'f1', cv = 5, verbose = 0, n_jobs = -1)
        grid_search.fit(X_train, y_train)

        # Log best parameters
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1-score: {grid_search.best_score_:.4f}")

        # Save best model
        model_path = f"{models_path}/lr_stemmed_tfidf.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        logger.info(f"Trained model saved to {model_path}")

        # Evaluate on validation set
        logger.info("Evaluating model on validation set")
        val_metrics = evaluate_model(grid_search.best_estimator_, X_val, y_val, "Tuned Logistic Regression (Stemmed + TF-IDF) (Validation)", safe_results = False)
        logger.info(f"Validation metrics: {val_metrics}")

        logger.info("Training and evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()
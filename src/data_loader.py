import pandas as pd
import logging
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from utils import load_config

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """
    Load train and test datasets from the specified directory.
    """
    try:
        # Load configuration
        config = load_config()

        train_path = config['paths']['train_path']
        test_path  = config['paths']['test_path']

        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"Training data shape: {train_df.shape}")

        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data shape: {test_df.shape}")

        # Check for missing values
        logger.info("Checking for missing values in training set")
        if train_df.isna().sum().sum() > 0:
            logger.warning(f"Missing values found in training set:\n{train_df.isna().sum()}")
        else:
            logger.info("No missing values in training set")

        logger.info("Checking for missing values in test set")
        if test_df.isna().sum().sum() > 0:
            logger.warning(f"Missing values found in test set:\n{test_df.isna().sum()}")
        else:
            logger.info("No missing values in test set")

        # Remove duplicates
        logger.info("Checking for duplicates in training set")
        train_df = train_df.drop_duplicates()
        logger.info(f"Training data shape after removing duplicates: {train_df.shape}")

        logger.info("Checking for duplicates in test set")
        test_df = test_df.drop_duplicates()
        logger.info(f"Test data shape after removing duplicates: {test_df.shape}")

        return train_df, test_df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    train_df, test_df = load_data()
    logger.info("Data loading completed successfully")
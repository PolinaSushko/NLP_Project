from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def vectorize_data(train_texts, test_texts=None, max_features=5000):
    """
    Vectorize text data using TF-IDF.
    """
    try:
        logger.info("Starting TF-IDF vectorization")
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = tfidf_vectorizer.fit_transform(train_texts)
        logger.info(f"Training data vectorized: {X_train.shape}")

        X_test = None
        if test_texts is not None:
            X_test = tfidf_vectorizer.transform(test_texts)
            logger.info(f"Test data vectorized: {X_test.shape}")

        return X_train, X_test, tfidf_vectorizer
    except Exception as e:
        logger.error(f"Error in vectorization: {e}")
        raise

def encode_labels(labels):
    """
    Encode labels using LabelEncoder.
    """
    try:
        logger.info("Encoding labels")
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        logger.info("Labels encoded successfully")
        return encoded_labels, encoder
    except Exception as e:
        logger.error(f"Error encoding labels: {e}")
        raise
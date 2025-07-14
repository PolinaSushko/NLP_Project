import re
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Setup logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")
    raise

def clean_text(text):
    """
    Basic text cleaning:
    1. Converting strings to lowercase
    2. Removing non-word and non-whitespace characters
    3. Removing digits
    """   
    try:
        logger.debug("Cleaning text")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise

def tokenize_text(text):
    """Tokenize text"""
    try:
        logger.debug("Tokenizing text")
        return word_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        raise
    
def remove_stopwords(tokens):
    """Remove stopwords"""
    try:
        logger.debug("Removing stopwords")
        stop_words = set(stopwords.words('english'))
        return [token for token in tokens if token not in stop_words]
    except Exception as e:
        logger.error(f"Error removing stopwords: {e}")
        raise
    
def stem_tokens(tokens):
    """Apply stemming"""
    try:
        logger.debug("Applying stemming")
        stemmer = SnowballStemmer('english')
        return [stemmer.stem(token) for token in tokens]
    except Exception as e:
        logger.error(f"Error stemming tokens: {e}")
        raise

def preprocess_text(text):
    """Complete text preprocessing pipeline"""
    try:
        #logger.info("Starting text preprocessing")
        cleaned = clean_text(text)
        tokens  = tokenize_text(cleaned)
        tokens  = remove_stopwords(tokens)
        tokens  = stem_tokens(tokens)
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise
import pandas as pd
import numpy as np
import re
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ======================= #
# Setup logging
# ======================= #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ======================= #
# Download NLTK Resources
# ======================= #
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


# ======================= #
# Text Cleaning Functions
# ======================= #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d', '', text)       # Remove digits
    return text


def tokenize_text(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


def stem_tokens(tokens):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text, use_stemming=False, use_lemmatization=False):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


# ======================= #
# Main processing function
# ======================= #
def main():
    # Config
    DATA_PATH_TRAIN = '../data/raw/train.csv'

    logger.info("Starting data processing...")

    # Step 1: Download necessary NLTK data
    download_nltk_resources()

    # Step 2: Load dataset
    logger.info("Loading dataset...")
    df_train = pd.read_csv(DATA_PATH_TRAIN)

    # Step 3: Preprocess text (Stemming and Lemmatization)
    logger.info("Preprocessing text with stemming...")
    df_stemmed = df_train.copy()
    df_stemmed['processed_review'] = df_stemmed['review'].apply(
        lambda x: preprocess_text(x, use_stemming=True)
    )

    logger.info("Preprocessing text with lemmatization...")
    df_lemmatized = df_train.copy()
    df_lemmatized['processed_review'] = df_lemmatized['review'].apply(
        lambda x: preprocess_text(x, use_lemmatization=True)
    )

    # Step 4: Encode labels
    logger.info("Encoding sentiment labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_train['sentiment'])

    # Step 5: Split dataset
    logger.info("Splitting data into train and validation sets...")
    X_stem = df_stemmed['processed_review']
    X_lem = df_lemmatized['processed_review']

    X_train_stem, X_val_stem, y_train, y_val = train_test_split(
        X_stem, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_lem, X_val_lem, _, _ = train_test_split(
        X_lem, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 6: Vectorization (CountVectorizer & TF-IDF)
    logger.info("Vectorizing text using CountVectorizer...")
    count_vec_stem = CountVectorizer(max_features=5000)
    count_vec_lem = CountVectorizer(max_features=5000)

    X_train_count_stem = count_vec_stem.fit_transform(X_train_stem)
    X_val_count_stem = count_vec_stem.transform(X_val_stem)

    X_train_count_lem = count_vec_lem.fit_transform(X_train_lem)
    X_val_count_lem = count_vec_lem.transform(X_val_lem)

    logger.info("Vectorizing text using TF-IDF...")
    tfidf_vec_stem = TfidfVectorizer(max_features=5000)
    tfidf_vec_lem = TfidfVectorizer(max_features=5000)

    X_train_tfidf_stem = tfidf_vec_stem.fit_transform(X_train_stem)
    X_val_tfidf_stem = tfidf_vec_stem.transform(X_val_stem)

    X_train_tfidf_lem = tfidf_vec_lem.fit_transform(X_train_lem)
    X_val_tfidf_lem = tfidf_vec_lem.transform(X_val_lem)

    logger.info("Vectorization complete. Shapes:")
    logger.info(f"Count (Stemmed): {X_train_count_stem.shape}")
    logger.info(f"TF-IDF (Stemmed): {X_train_tfidf_stem.shape}")
    logger.info(f"Count (Lemmatized): {X_train_count_lem.shape}")
    logger.info(f"TF-IDF (Lemmatized): {X_train_tfidf_lem.shape}")

    # Final report
    logger.info("Data processing pipeline completed successfully.")


# ======================= #
# Entry point
# ======================= #
if __name__ == "__main__":
    main()

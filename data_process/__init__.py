# train_sentiment_model.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ----------------------------
# Preprocessing Functions
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)

# ----------------------------
# Training Pipeline
# ----------------------------
def train_model(df):
    # Preprocess reviews
    df['processed_review'] = df['review'].apply(preprocess_text)

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df['processed_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_val_vec = tfidf_vectorizer.transform(X_val)

    # Train Logistic Regression
    model = LogisticRegression(C=10, solver='liblinear', penalty='l2', max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_val_vec)
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logreg_model.joblib")
    joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(label_encoder, "models/label_encoder.joblib")
    print("✅ Model, vectorizer, and encoder saved to 'models/'")

# ----------------------------
# Main Entry Point
# ----------------------------
def main():
    # Load pre-cleaned dataset
    data_path = "../data/raw/train.csv"
    df = pd.read_csv(data_path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Train
    train_model(df)

if __name__ == "__main__":
    main()

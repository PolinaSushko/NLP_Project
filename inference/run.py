#!/usr/bin/env python3
"""
Sentiment inference script for the Logistic-Regression + TF-IDF model.

Usage examples
--------------
# Single text string
python sentiment_inference.py --model_dir models \
       --text "I absolutely loved this!"

# Batch file (one review per line)
python sentiment_inference.py --model_dir models \
       --input_file reviews.txt
"""

import argparse
import joblib
import os
import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from typing import List

# Ensure NLTK resources are available (no-ops if already downloaded)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# --------------------------------------------------------------------
# Pre-processing utilities (identical to training script)
# --------------------------------------------------------------------
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = SnowballStemmer("english")

def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

def _preprocess(text: str) -> str:
    tokens = word_tokenize(_clean_text(text))
    tokens = [t for t in tokens if t not in _STOPWORDS]
    tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

# --------------------------------------------------------------------
# Core inference class
# --------------------------------------------------------------------
class SentimentPredictor:
    """Lazy loader & wrapper around the trained sklearn pipeline."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        try:
            self.model = joblib.load(os.path.join(model_dir, "logreg_model.joblib"))
            self.vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))
            self.label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        except FileNotFoundError as exc:
            sys.exit(f"[ERROR] Expected model files not found in {model_dir}: {exc}")

    def predict(self, texts: List[str]) -> List[str]:
        processed = [_preprocess(t) for t in texts]
        X = self.vectorizer.transform(processed)
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds).tolist()

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment inference CLI")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing logreg_model.joblib, tfidf_vectorizer.joblib, label_encoder.joblib")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single review string to classify")
    group.add_argument("--input_file", help="Path to a text file (one review per line)")
    return parser.parse_args()

def main():
    args = parse_args()
    predictor = SentimentPredictor(args.model_dir)

    if args.text:
        reviews = [args.text]
    else:
        if not os.path.isfile(args.input_file):
            sys.exit(f"[ERROR] File not found: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as f:
            reviews = [line.strip() for line in f if line.strip()]

    predictions = predictor.predict(reviews)
    for review, label in zip(reviews, predictions):
        print(f"\"{review[:60]}...\" → {label}")

if __name__ == "__main__":
    main()

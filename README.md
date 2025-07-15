# Sentiment Analysis Project

This repository contains a comprehensive sentiment analysis pipeline for classifying text reviews as positive or negative. The project includes both Data Science (DS) and Machine Learning Engineering (MLE) components, with scripts for data preprocessing, feature engineering, model training, and inference. The pipeline compares traditional machine learning models (Logistic Regression, Random Forest, Naive Bayes) with a transformer-based model (DistilBERT) and is fully containerized using Docker for reproducibility.

## Project Structure

```
NLP_PROJECT/
|-- data/                     # Input data 
|   |-- raw/                  # train.csv, test.csv
|-- notebooks/                # Jupyter notebooks for DS exploration
|-- src/                      # Source code
|   |-- train/                # Training scripts
|   |   |-- train.py
|   |   |-- Dockerfile
|   |-- inference/            # Inference scripts
|   |   |-- run_inference.py
|   |   |-- Dockerfile
|   |-- data_loader.py
|   |-- data_processing.py
|   |-- feature_engineering.py
|   |-- evaluation.py
|-- outputs/                  # Outputs
|   |-- models/               # Trained models
|   |-- predictions/          # Inference results
|   |-- figures/              # Plots
|-- README.md
|-- requirements.txt
|-- utils.py
|-- settings.json
|-- .gitignore
```

## Data Science (DS) Part

### Exploratory Data Analysis (EDA)

The dataset consists of movie reviews labeled as positive or negative sentiments, split into training and test sets.
- **Dataset Overview:**
    - Training set: ~40,000 reviews 
    - Test set: ~10,000 reviews
    - No missing values in either dataset.
    - Balanced sentiment distribution: ~50% positive, ~50% negative in both sets.
- **Text Length Analysis:**
    - Average review length: ~1,311 characters.
    - Distribution: Right-skewed, with most reviews between 700 and 1,596 characters.
    - Word Clouds:
        - Positive reviews: Frequent words include "great," "good," "love," "excellent."
        - Negative reviews: Frequent words include "bad," "poor," "disappointing," "worst."
- The dataset is balanced, clean, and suitable for sentiment analysis. The text length distribution suggests most reviews are concise, and word clouds highlight sentiment-specific vocabulary.

### Feature Engineering
The pipeline implements comprehensive text preprocessing and vectorization techniques:
- **Tokenization:**
    - Used NLTK's word_tokenize to split text into tokens.
    - Ensures consistent word-level processing.
- **Stop-Words Filtering:**
    - Removed common English stop-words (e.g., "the," "is") using NLTK's stopword list.
    - Reduces noise and focuses on meaningful words.
- **Stemming vs. Lemmatization:**
    - *Stemming (SnowballStemmer):* 
        - Reduces words to their root form (e.g., "movies" → "movi," "warning" → "warn"). Faster but can produce non-words (e.g., "little" → "littl").
    - *Lemmatization (WordNetLemmatizer):*
        -  Returns dictionary forms (e.g., "movies" → "movie," "warning" → "warning"). Preserves semantic meaning, more readable.
    - *Observation:* Lemmatization maintains word integrity, while stemming reduces vocabulary size more aggressively.
- **Vectorization:**
    - *Bag of Words (BoW):* Uses CountVectorizer to create a frequency-based representation (max 5,000 features).
    - *TF-IDF:* Uses TfidfVectorizer to weigh terms by importance (max 5,000 features). Emphasizes distinctive words for sentiment analysis.
    - *Comparison:* TF-IDF outperforms BoW by focusing on discriminative terms, improving model performance.
- *Conclusion:* Lemmatization + TF-IDF provides a balanced approach, preserving meaning while emphasizing important terms. Stemming + TF-IDF also performs well, particularly for Logistic Regression.

### Modeling
Three traditional machine learning models (Logistic Regression, Random Forest, Naive Bayes) and one transformer-based model (DistilBERT) were evaluated:
- Models were tested on four configurations: Stemmed + BoW, Lemmatized + BoW, Stemmed + TF-IDF, Lemmatized + TF-IDF.
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Hyperparameter tuning was performed for traditional models using RandomizedSearchCV.

**Results on Validation Set:**
- *Tuned Logistic Regression (Stemmed + TF-IDF):*
    - Accuracy: 0.885, F1-Score: 0.887, ROC-AUC: 0.952
    - Best performer due to its alignment with TF-IDF features.
- *Tuned Random Forest (Lemmatized + TF-IDF):*
    - Accuracy: 0.846, F1-Score: 0.847, ROC-AUC: 0.922
    - Improved after tuning but less effective than Logistic Regression.
- *Tuned Naive Bayes (Stemmed + TF-IDF):*
    - Accuracy: 0.843, F1-Score: 0.844, ROC-AUC: 0.924
    - Benefited from TF-IDF and tuning.
- *DistilBERT:*
    - Accuracy: 0.883, F1-Score: 0.883, ROC-AUC: 0.952
    - Strong ROC-AUC but slightly lower accuracy/F1 than Logistic Regression.

**Test Set Results:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Tuned Logistic Regression (Stemmed + TF-IDF) | 0.925 | 0.914 | 0.935 | 0.924 | 0.976 |
| DistilBERT | 0.883 | 0.891 | 0.876 | 0.883 | 0.954 |

**Best Model Selection:**
- **Tuned Logistic Regression (Stemmed + TF-IDF)** was selected as the best model due to its superior performance across all metrics on the test set (Accuracy ≥ 0.85, F1-Score: 0.924).
- **Reasoning:** Logistic Regression aligns well with TF-IDF's linear feature space, is computationally efficient, and outperforms the more complex DistilBERT, possibly due to BERT's early stopping or suboptimal hyperparameters.

**Conclusion:** The tuned Logistic Regression model with Stemmed + TF-IDF features achieves the best balance of performance, simplicity, and efficiency.

### Potential Business Applications
**Customer Feedback Analysis:**
- Automatically classify customer reviews (e.g., for movies, products, or services) to gauge sentiment and identify areas for improvement.
- It enables businesses to prioritize customer satisfaction and address negative feedback promptly.

**Market Research:**
- Analyze social media or review platforms to understand consumer sentiment toward brands or products.
- It informs marketing strategies and product development by identifying trends and preferences.

**Content Moderation:**
- Flag negative or toxic reviews for further review, improving platform user experience.
- It enhances brand reputation and user trust.

Sentiment analysis provides actionable insights for businesses, driving customer-centric decisions and operational efficiency.

## Machine Learning Engineering (MLE) Part

### Quickstart 
**1. Install Docker:** Ensure Docker is installed on your system.

**2. Clone the Repository:**
```
git clone https://github.com/PolinaSushko/NLP_project.git
cd NLP_PROJECT
```

**3. Prepare Data:**
```
python src/data_loader.py
```

**4. Directory Setup:**

Ensure the outputs/ directory exists for storing models, predictions, and figures:
```
mkdir -p outputs/models outputs/predictions outputs/figures
```

### Training with Docker
The training process preprocesses data, trains the tuned Logistic Regression model (Stemmed + TF-IDF), and evaluates it on the validation set. The trained model is saved to the outputs/ directory.

**1. Build the Training Docker Image:**
```
docker build -t sentiment-train -f src/train/Dockerfile .
```

**2. Run the Training Container:**

Mount the data/ and outputs/ directories to persist data and results.

The container automatically runs train.py, trains the model, saves it to outputs/models/lr_stemmed_tfidf.pkl.
```
docker run --rm -v $(pwd)/../../data:/app/data -v $(pwd)/../../outputs:/app/outputs sentiment-train
```

**3. Outputs:**
- Model: outputs/models/lr_stemmed_tfidf.pkl
- Metrics are printed in console.

### Inference with Docker
The inference process loads the trained Logistic Regression model and generates predictions on the test set, saving testing plots to the outputs/ directory and results to outputs/predictions/.

**1. Build the Inference Docker Image:**
```
docker build -t sentiment-inference . -f src/inference/Dockerfile .
```

**2. Run the Inference Container:**

Mount the data/ and outputs/ directories to access the test data and trained model.

The container runs run.py, generates predictions, and saves them to outputs/predictions/predictions.csv.

Testing metrics and plots are also generated.
```
docker run --rm -v $(pwd)/../../data:/app/data -v $(pwd)/../../outputs:/app/outputs sentiment-inference
```

**3. Outputs:**
- Predictions: outputs/predictions/predictions.csv (contains review, true label, predicted label)
- Plots: Confusion matrix and ROC curve in outputs/figures/
- Metrics are printed in console and saved in plots

## Requirements

The requirements.txt file includes all necessary dependencies:
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud
- torch
- transformers
- datasets
- evaluate
- joblib

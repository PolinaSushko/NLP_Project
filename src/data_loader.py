import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from src.exception import CustomException
from src.logger import logging

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

class DataLoader:
    def __init__(self, num_words = 10000, max_text_len = 100):
        self.NUM_WORDS = num_words
        self.MAX_TEXT_LEN = max_text_len
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.tokenizer = Tokenizer(num_words=self.NUM_WORDS)

    def load_data(self, train_path, test_path):
        logging.info("Loading data from %s and %s", train_path, test_path)

        try:
            train_data = pd.read_csv(train_path)
            test_data  = pd.read_csv(test_path)

            logging.info("Data loaded successfully.")

        except Exception as e:
            logging.error("Error in data loading", exc_info = True)
            raise e

        return train_data, test_data

    def preprocess_text(self, text):
        """
        Preprocesses input text by applying a series of transformations to clean and standardize it.

        Steps involved in preprocessing:
        1. Converts text to lowercase to ensure uniformity.
        2. Tokenizes the text into individual words (tokens).
        3. Removes punctuation, non-alphanumeric characters, and stop words.
        4. Applies stemming to reduce words to their root form.
        5. Joins the tokens back into a single string.

        Parameters:
        text (str): The input text string to be preprocessed.

        Returns:
        preprocessed_text (str): A preprocessed, cleaned, and stemmed version of the input text.
        """
        logging.info("Preprocessing text")

        try:
            text   = text.lower()
            tokens = word_tokenize(text)
            tokens = [
                self.stemmer.stem(word) 
                for word in tokens if word.isalnum() and word not in string.punctuation and word not in self.stop_words
            ]

            preprocessed_text = ' '.join(tokens)

            logging.info("Text preprocessing completed")

            return preprocessed_text

        except Exception as e:
            logging.error("Error in text preprocessing", exc_info = True)
            raise e

    def tokenize_text(self, text):
        logging.info("Tokenizing text data and mapping sentiment labels")

        try:
            text.loc[:, 'sentiment'] = text['sentiment'].map({'negative' : 0, 
                                                            'positive' : 1})
            
            review    = text['processed_review']
            sentiment = text['sentiment']

            self.tokenizer.fit_on_texts(review)

            sequences = self.tokenizer.texts_to_sequences(review)

            logging.info("Tokenization and label mapping completed.")

            return sequences, sentiment
        
        except Exception as e:
            logging.error("Error in text tokenizing", exc_info = True)
            raise e

    def apply_preprocess(self, train_path, test_path):
        logging.info("Starting data preprocessing")

        try:
            train_data, test_data = self.load_data(train_path, test_path)

            train_data.loc[:, 'processed_review'] = train_data['review'].apply(self.preprocess_text)
            test_data.loc[:, 'processed_review'] = test_data['review'].apply(self.preprocess_text)

            train_sequences, train_sentiment = self.tokenize_text(train_data)
            test_sequences, test_sentiment   = self.tokenize_text(test_data)

            logging.info("Preprocessing and tokenization complete")

            return train_sequences, train_sentiment, test_sequences, test_sentiment
        
        except Exception as e:
            logging.error("Error in preprocessor applying", exc_info = True)
            raise e

    def create_train_test_ds(self, train_path, test_path):
        logging.info("Creating training and test datasets with padding")

        try:
            train_sequences, train_sentiment, test_sequences, test_sentiment = self.apply_preprocess(train_path, test_path)

            X_train = pad_sequences(train_sequences, maxlen = self.MAX_TEXT_LEN)
            y_train = train_sentiment.copy()

            X_test = pad_sequences(test_sequences, maxlen = self.MAX_TEXT_LEN)
            y_test = test_sentiment.copy()

            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)
            X_test  = np.array(X_test, dtype=np.float32)
            y_test  = np.array(y_test, dtype=np.float32)

            # Flatten X_train and X_test for saving into CSV
            train_data = np.column_stack((X_train, y_train)) # Combine features and labels
            test_data  = np.column_stack((X_test, y_test))  # Combine features and labels

            # Save the combined data into CSV
            pd.DataFrame(train_data).to_csv("E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/processed/train_data.csv", index=False, header=False)
            pd.DataFrame(test_data).to_csv("E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/processed/test_data.csv", index=False, header=False)

            logging.info("Datasets created")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logging.error("Error in creating in train and test sets", exc_info = True)
            raise e

if __name__ == "__main__":
    train_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/raw/train.csv'
    test_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/raw/test.csv'

    data_loader = DataLoader(num_words = 10000, max_text_len = 100)

    X_train, y_train, X_test, y_test = data_loader.create_train_test_ds(train_path, test_path)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

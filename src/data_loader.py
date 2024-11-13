import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string
from keras_preprocessing.text import Tokenizer

# Download necessary resources
nltk.download('punkt')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Create a stemmer for stemming words
stemmer = SnowballStemmer('english')

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    return train_data, test_data

def preprocess_text(text):
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
    text   = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in string.punctuation and word not in stop_words]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def apply_preprocess(train_path, test_path):
    train_data, test_data = load_data(train_path, test_path)

    train_data.loc[:, 'processed_review'] = train_data['review'].apply(preprocess_text)
    test_data.loc[:, 'processed_review'] = test_data['review'].apply(preprocess_text)

    return train_data, test_data

def tokenize_text(text):
    review    = text['processed_review']
    sentiment = text['sentiment']

    NUM_WORDS    = 10000
    MAX_TEXT_LEN = 100

    tokenizer = Tokenizer(num_words = NUM_WORDS)
    tokenizer.fit_on_texts(review)

    sequences = tokenizer.texts_to_sequences(review)

    a.loc[:, 'sentiment'] = a['sentiment'].map({'negative' : 0, 
                 'positive' : 1})

#a, b = apply_preprocess('E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/train.csv', 
#                        'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/test.csv')

#print(a.head())
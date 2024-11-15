import sys
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.exception import CustomException
from src.logger import logging
from src.data_loader import DataLoader
from utils import save_object, evaluate_model

class ModelTrainer:
    """
    A class for building, training, and saving an LSTM-based model.

    Attributes:
    - input_dim (int): The size of the vocabulary for the embedding layer.
    - embedding_dim (int): The dimension of the embedding vectors.
    - input_length (int): The length of the input sequences.

    Methods:
    - build_model: Builds and compiles an LSTM model for binary classification.
    - train: Trains the model on given training data, evaluates its accuracy, and saves the trained model.
    """
    def __init__(self, input_dim = 10000, embedding_dim = 64, input_length = 100):
        """
        Initializes the ModelTrainer with specified embedding and input parameters.

        Parameters:
        - input_dim (int): Vocabulary size for the embedding layer (default: 10000).
        - embedding_dim (int): Dimension of the embedding vector (default: 64).
        - input_length (int): Length of each input sequence (default: 100).
        """
        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim
        self.input_length  = input_length

    def build_model(self):
        """
        Builds and compiles an LSTM model for binary classification with the following architecture:
        - Embedding layer based on input dimensions.
        - Three LSTM layers with increasing units and dropout for regularization.
        - Final Dense layer with sigmoid activation for binary output.

        Returns:
        - model (keras.Sequential): The compiled Keras model ready for training.
        """
        logging.info("Building the model")

        try:
            model = Sequential()
            model.add(Embedding(self.input_dim, self.embedding_dim))
            model.add(LSTM(32, return_sequences = True))
            model.add(Dropout(0.5))  
            model.add(LSTM(64, return_sequences = True))
            model.add(Dropout(0.5))
            model.add(LSTM(128))
            model.add(Dense(1, activation = 'sigmoid'))

            model.compile(metrics = ['accuracy'], loss = 'binary_crossentropy', optimizer = 'Adam')

            logging.info("Model built successfully")
            return model

        except Exception as e:
            raise CustomException(e, sys)

    def train(self, X_train, y_train, X_test, y_test):
        """
        Trains the built LSTM model on the provided training data, evaluates it, and saves the model.

        Parameters:
        - X_train (np.array): Training feature data.
        - y_train (np.array): Training labels.
        - X_test (np.array): Testing feature data for evaluation.
        - y_test (np.array): Testing labels for evaluation.

        Process:
        - Initializes early stopping to prevent overfitting.
        - Trains the model using the `evaluate_model` function with early stopping.
        - Logs the training progress and model accuracy.
        - Saves the trained model as 'model.pkl' in the 'outputs/models/' directory.

        Returns:
        - model_accuracy (float): The accuracy of the model on the testing data.
        """    
        logging.info("Starting model training")

        model = self.build_model()

        try:
            early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
            logging.info("Early stopping callback initialized")

            logging.info("Training the model")
            model_accuracy = evaluate_model(X_train, y_train, X_test, y_test, model, early_stopping)
            logging.info(f"Model evaluation completed with accuracy: {model_accuracy}")

            logging.info("Saving the model")
            save_object(file_path = 'outputs/models/model.pkl',
                        obj = model)
            logging.info(f"Model saved")

            return model_accuracy
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'

    data_loader = DataLoader(num_words = 10000, max_text_len = 100)
    X_train, y_train, X_test, y_test = data_loader.create_train_test_processed_ds(train_path, test_path)

    model_trainer = ModelTrainer(input_dim = 10000, embedding_dim = 64, input_length = 100)
    accuracy = model_trainer.train(X_train, y_train, X_test, y_test)

    print(accuracy)

import sys
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.exception import CustomException
from src.logger import logging
from src.data_loader import DataLoader
from utils import save_object, evaluate_model

class ModelTrainer:
    def __init__(self, input_dim = 10000, embedding_dim = 64, input_length = 100):
        self.input_dim     = input_dim
        self.embedding_dim = embedding_dim
        self.input_length  = input_length

    def build_model(self):
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
        logging.info("Starting model training")

        model = self.build_model()

        try:
            early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)
            logging.info("Early stopping callback initialized")

            logging.info("Training the model")
            model_accuracy = evaluate_model(X_train, y_train, X_test, y_test, model, early_stopping)
            logging.info(f"Model evaluation completed with accuracy: {model_accuracy}")

            logging.info("Saving the model")
            save_object(file_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/outputs/models/model.pkl',
                        obj = model)
            logging.info(f"Model saved")

            return model_accuracy
        
        except Exception as e:
            raise CustomException(e, sys)

#if __name__ == "__main__":
#    train_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/raw/train.csv'
#    test_path = 'E:/Work Folder/0_Polya/My/EPAM/Introduction to Data Science Program/NLP_Project/data/raw/test.csv'

#    data_loader = DataLoader(num_words = 10000, max_text_len = 100)
#    X_train, y_train, X_test, y_test = data_loader.create_train_test_ds(train_path, test_path)

#    model_trainer = ModelTrainer(input_dim = 10000, embedding_dim = 64, input_length = 100)
#    accuracy = model_trainer.train(X_train, y_train, X_test, y_test)

#    print(accuracy)

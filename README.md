## NLP Classification Project: Sentiment Analysis with Machine Learning

### Overview
This project focuses on building a machine learning model to classify the sentiment of text data. Using various machine learning algorithms, the goal is to predict whether a text expresses positive or negative sentiment. The dataset used consists of 50,000 polar movie reviews.

### Dataset Source
- **Description**: The dataset contains text entries labeled with sentiment categories such as positive and negative. The project applies Natural Language Processing (NLP) techniques to clean and prepare the data for classification tasks.

### Key Features:
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization, and vectorization (e.g., TF-IDF) were applied to prepare text data for model input.
- **Custom Transformers**: Feature engineering involved creating custom transformers to enhance the dataset, such as calculating word frequency and extracting key linguistic features.

### Modeling
Several machine learning algorithms were trained and compared:
1. **Logistic Regression**
2. **Random Forest**
3. **Decision Tree**
4. **XGBoost**
5. **LSTM**

### Model Performance
Each model's performance was evaluated:
- **Logistic Regression** performs well with negative classes, achieving an F1-score of 0.62. However, its F1-score for positive classes is slightly lower at 0.46.
- **Decision Tree** demonstrates the lowest performance metrics, with an F1-score across all classes ranging from approximately 0.53 to 0.56.
- **Random Forest** performs well with negative classes, achieving an F1-score of 0.63, which is the highest among all models.
- **Gradient Boosting** performs poorly with negative classes, achieving an F1-score of 0.59.
- **LSTM** performs decently with final accuracy of 0.8735.

### Hyperparameter Tuning
For LSTM, hyperparameter tuning was performed using **Randomized Search** to optimize model performance:
- **Best Accuracy**: 0.87 after tuning.

### Project Structure
This project has a modular structure, where each folder has a specific duty.
```
├── data/  
|   ├── raw                          # Data files used for training and inference
│   ├── processed                     
├── notebook/                        # Exploratory Data Analysis (EDA) and prototyping the model training process notebook             
│   ├── nlp_project.ipynb    
├── outputs/                         # Folder where trained models and predictions are stored
|   ├── models                        
│   ├── predictions                    
├── src/                             # Source code directory containing core project modules
│   ├── inference/                   # Scripts and Dockerfiles used for inference
│   |    ├── Dockerfile 
│   |    ├── run_inference.py
│   ├── train/                       # Scripts and Dockerfiles used for training
|   |    ├── Dockerfile                 
│   |    ├── train.py  
|   ├── data_loader.py               # Scripts used for loading, preprocessing, and tokenizing data
|   ├── exception.py                 # Custom exception handling for error tracking
|   ├── logger.py                    # Logging setup for monitoring and debugging
├── utils.py                         # Utility functions for data processing and modeling tasks
├── .gitignore  
├── README.md   
├── requirements.txt                 # List of required packages for the project          
└── setup.py                         # Setup script for packaging and distribution of the project
```

### Installation
Clone this repository:
```
git clone https://github.com/PolinaSushko/NLP_Project.git
cd NLP_Project
```

### Training
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script src/train/train.py.

To train the model using Docker:
1. Build the training Docker image. If the built is successfully done, it will automatically train the model:
```
docker build -f ./src/train/Dockerfile -t training_image .
```
2. You may run the container with the following parameters to ensure that the trained model is here:
```
docker run -it training_image /bin/bash
```
3. Then, move the trained model from the directory inside the Docker container /app/models to the local machine using:
```
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```
Replace <container_id> with your running Docker container ID and <model_name>.pickle with your model's name.

- Alternatively, the train.py script can also be run locally as follows:
```
python -m src.train.train
```

### Inference
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in src/inference/run_inference.py.

To run the inference using Docker, use the following commands:
1. Build the inference Docker image:
```
docker build -f ./src/inference/Dockerfile -t inference_image .
```
2. Run the inference with the attached terminal using the following command:
```
docker run -it inference_image /bin/bash
```
- Alternatively, you can also run the inference script locally:
```
python -m src.inference.run_inference
```
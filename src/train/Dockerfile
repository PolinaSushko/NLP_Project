# Dockerfile for training
# Start from a base Python image
FROM python:3.11

# Changing the path to the working directory in the container
WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy the code itself
COPY src/train /app/training

# Copy data_loader.py
COPY src/data_loader.py /app

# Copy exception.py
COPY src/exception.py /app

# Copy logger.py
COPY src/logger.py /app

# Copy utils.py
COPY utils.py /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Train the model
RUN python3 src/train/train.py

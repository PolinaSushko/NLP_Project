# Dockerfile for inference
# Start from a base Python image
FROM python:3.11

WORKDIR /app

# Copy data folder to Docker
COPY data/ /app/data

# Copy models folder to Docker
COPY outputs/models /app/models

# Copy the code itself
COPY src/inference /app/inference

# Copy usefull scripts
COPY src/data_loader.py /app
COPY src/exception.py /app
COPY src/logger.py /app
COPY utils.py /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Run inference
RUN python3 src/inference/run_inference.py
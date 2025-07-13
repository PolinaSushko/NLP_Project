FROM python:3.10

# Set default config file (optional in your case)
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script
COPY train_sentiment_model.py .

# Copy raw data directory
COPY data/ /app/data

# Copy config file if used
COPY ${CONF_PATH} /app

# Create necessary directories
RUN mkdir -p /app/models

# Run the training script
CMD ["python", "train.py"]

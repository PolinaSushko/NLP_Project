FROM python:3.10

# Allow the model directory to be supplied at build time if desired
ARG model_dir=models
ENV MODEL_DIR=${model_dir}

WORKDIR /app

# Install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY sentiment_inference.py .

# Copy model artifacts into the container
# (substitute your own path if you mount a volume instead)
COPY ${MODEL_DIR} /app/models

# Default command: allow overriding CLI args at `docker run`
ENTRYPOINT ["python", "run.py"]
CMD ["--model_dir", "/app/models", "--text", "This is amazing!"]

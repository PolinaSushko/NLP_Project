FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=../../settings.json
ENV CONF_PATH=${settings_name}

# Set working directory
WORKDIR /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Copy utils.py 
COPY utils.py .

# Copy data and models directories (will be mounted)
COPY data/ data/
COPY outputs/models/ outputs/models/

# Create output directories
RUN mkdir -p outputs/predictions outputs/figures

# Copy config file if used
COPY ${CONF_PATH} /app

# Set working directory to inference
WORKDIR /app/src/inference

# Run the training script
CMD ["python", "run.py"]

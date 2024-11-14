#FROM python:3.11

#WORKDIR /app

#COPY . .

#RUN pip install -r requirements.txt

#CMD ["python", "train.py"]

# Stage 1: Build dependencies
FROM python:3.11 AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.11

WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=build /app .

CMD ["python", "train.py"]

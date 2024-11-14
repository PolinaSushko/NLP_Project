#FROM python:3.11

#WORKDIR /app

#COPY . .

#RUN pip install --no-cache-dir -r requirements.txt

#CMD ["python", "train.py"]

FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]

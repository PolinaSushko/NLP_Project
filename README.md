# NLP_Project

Binary sentiment classification on movie reviews.

## Quickstart

### Training

```bash
docker build -t sentiment-train -f training/Dockerfile .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models sentiment-train
```

### Inference

```bash
docker build -t sentiment-inf -f inference/Dockerfile .
docker run --rm \\
    -v $(pwd)/data:/app/data \\
    -v $(pwd)/models:/app/models \\
    -v $(pwd)/inference_results:/app/inference_results \\
    sentiment-inf --input_file data/raw/test.csv
```

## Structure

```
sentiment-classifier/
├── data/
│   └── raw/
│       ├── test.csv
│       └── train.csv
├── data_process/
│   ├── __init__.py
│   └── data_processing.py
├── inference/
│   ├── Dockerfile
│   └── run.py
├── models/
│   └── .gitkeep
├── notebooks/
│   └── nlp_project.ipynb
├── src/
│   ├── data_loader.py
│   └── training/
├── .gitignore
├── README.md
└── requirements.txt

```
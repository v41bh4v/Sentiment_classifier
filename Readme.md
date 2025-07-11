# Crypto Sentiment Classifier

This repository contains a minimal NLP sentiment classifier for cryptocurrency-related Reddit comments, with a REST API for predictions.

## Project Structure

- `eda.py`: Exploratory Data Analysis.
- `model_pipeline.py`: Training and evaluation pipeline.
- `api.py`: REST API for serving predictions.
- `crypto_currency_sentiment_dataset.csv`: Input dataset.

## How to Use

Install dependencies

```bash
pip install -r requirements.txt


## Train the model

python model_pipeline.py

##  Run the API

uvicorn api:app --reload

## Test API
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"comment\": \"I love Bitcoin and think it's going to the moon!\"}"

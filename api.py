# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("sentiment_model.joblib")

# Define app
app = FastAPI(title="Crypto Sentiment API")

# Define input schema
class CommentInput(BaseModel):
    comment: str

# Define route
@app.post("/predict")
def predict_sentiment(data: CommentInput):
    prediction = model.predict([data.comment])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment}

# model_pipeline.py

import pandas as pd
import numpy as np
import joblib
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("crypto_currency_sentiment_dataset.csv")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Apply cleaning
df['Comment_clean'] = df['Comment'].apply(clean_text)

# Encode target
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['Comment_clean'], df['Sentiment'], test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "sentiment_model.joblib")

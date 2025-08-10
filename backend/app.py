# backend/app.py

import os
from fastapi import FastAPI, HTTPException
import joblib
import uvicorn
from pydantic import BaseModel
import string
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Configuration
origins = [
    "https://your-frontend-domain.com",  # Replace with your frontend domain
    "http://localhost:3000",             # For local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the specified origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Advanced text preprocessing without spacy dependency

# Load the best available model and vectorizer
try:
    # Try to load advanced model first
    if os.path.exists("advanced_sentiment_model.joblib") and os.path.exists("advanced_sentiment_vectorizer.joblib"):
        model = joblib.load("advanced_sentiment_model.joblib")
        vectorizer = joblib.load("advanced_sentiment_vectorizer.joblib")
        print("Advanced ensemble model loaded successfully.")
        model_type = "advanced"
    else:
        # Fall back to legacy model
        model = joblib.load("sentiment_nb_model.joblib")
        vectorizer = joblib.load("sentiment_vectorizer.joblib")
        print("Legacy model loaded successfully.")
        model_type = "legacy"
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

class Review(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    """
    Advanced text preprocessing without spacy dependency.
    """
    try:
        if not isinstance(text, str):
            return ""
        
        # Lowercase the text
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return ""

@app.post("/predict")
def predict_sentiment(review: Review):
    try:
        processed_text = preprocess_text(review.text)
        features = vectorizer.transform([processed_text])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            },
            "model_type": model_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

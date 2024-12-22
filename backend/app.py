# backend/app.py

from fastapi import FastAPI, HTTPException
import joblib
import uvicorn
from pydantic import BaseModel
import string
import spacy
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

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    print("spaCy model loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    raise e

# Load the pre-trained model and vectorizer once at startup
try:
    model = joblib.load("sentiment_nb_model.joblib")
    vectorizer = joblib.load("sentiment_vectorizer.joblib")
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

class Review(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Lemmatizing
    """
    try:
        if not isinstance(text, str):
            return ""
        
        # Lowercase the text
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        
        # Join tokens back to string
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return ""

@app.post("/predict")
def predict_sentiment(review: Review):
    try:
        processed_text = preprocess_text(review.text)
        features = vectorizer.transform([processed_text])
        prediction = model.predict(features)[0]
        return {"sentiment": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

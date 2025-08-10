# backend/enhanced_app.py - Advanced sentiment analysis API with improved features

from fastapi import FastAPI, HTTPException
import joblib
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional
import string
import json
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Sentiment Analysis API",
    description="Enhanced sentiment analysis with ensemble models and confidence scores",
    version="2.0.0"
)

# CORS Configuration
origins = [
    "https://your-frontend-domain.com",  # Replace with your frontend domain
    "http://localhost:3000",             # For local development
    "http://127.0.0.1:3000",            # Alternative local address
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class Review(BaseModel):
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)

class BatchReview(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", max_items=100)

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment: 'pos' or 'neg'")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    probabilities: dict = Field(..., description="Probability scores for each class")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of sentiment predictions")
    summary: dict = Field(..., description="Summary statistics")

class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    training_accuracy: Optional[float]
    cross_validation_score: Optional[float]

# Global variables for models
advanced_model = None
advanced_vectorizer = None
legacy_model = None  # Keep legacy model for comparison
legacy_vectorizer = None
model_metrics = None

def advanced_preprocess_text(text: str) -> str:
    """
    Advanced text preprocessing function.
    """
    try:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    except Exception as e:
        logger.error(f"Error in preprocessing text: {e}")
        return ""

def load_models():
    """Load all available models."""
    global advanced_model, advanced_vectorizer, legacy_model, legacy_vectorizer, model_metrics
    
    try:
        # Load advanced model
        if os.path.exists("advanced_sentiment_model.joblib") and os.path.exists("advanced_sentiment_vectorizer.joblib"):
            advanced_model = joblib.load("advanced_sentiment_model.joblib")
            advanced_vectorizer = joblib.load("advanced_sentiment_vectorizer.joblib")
            logger.info("Advanced ensemble model loaded successfully.")
            
            # Load model metrics if available
            if os.path.exists("model_metrics.json"):
                with open("model_metrics.json", 'r') as f:
                    model_metrics = json.load(f)
        else:
            logger.warning("Advanced model not found, will use legacy model.")
        
        # Load legacy model as backup
        if os.path.exists("sentiment_nb_model.joblib") and os.path.exists("sentiment_vectorizer.joblib"):
            legacy_model = joblib.load("sentiment_nb_model.joblib")
            legacy_vectorizer = joblib.load("sentiment_vectorizer.joblib")
            logger.info("Legacy model loaded successfully.")
        
        if not advanced_model and not legacy_model:
            raise Exception("No models found!")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

def predict_sentiment_advanced(text: str) -> dict:
    """
    Predict sentiment using the advanced ensemble model.
    """
    try:
        # Use advanced model if available, otherwise fall back to legacy
        if advanced_model and advanced_vectorizer:
            processed_text = advanced_preprocess_text(text)
            features = advanced_vectorizer.transform([processed_text])
            prediction = advanced_model.predict(features)[0]
            probabilities = advanced_model.predict_proba(features)[0]
            model_used = "ensemble"
        elif legacy_model and legacy_vectorizer:
            # For legacy model, keep original preprocessing
            processed_text = text.lower().translate(str.maketrans('', '', string.punctuation))
            features = legacy_vectorizer.transform([processed_text])
            prediction = legacy_model.predict(features)[0]
            probabilities = legacy_model.predict_proba(features)[0]
            model_used = "legacy"
        else:
            raise Exception("No model available for prediction")
        
        # Calculate confidence as max probability
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            "negative": float(probabilities[0]) if prediction == 'neg' else float(probabilities[1]),
            "positive": float(probabilities[1]) if prediction == 'pos' else float(probabilities[0])
        }
        
        # Ensure negative comes first in array for consistency
        if model_used == "ensemble":
            prob_dict = {"negative": float(probabilities[0]), "positive": float(probabilities[1])}
        
        return {
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_used
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Load models at startup
@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Advanced Sentiment Analysis API",
        "version": "2.0.0",
        "features": [
            "Ensemble model with Naive Bayes, Logistic Regression, and Random Forest",
            "Confidence scores and probability distributions",
            "Batch processing capabilities",
            "Advanced text preprocessing",
            "Model comparison and metrics"
        ]
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(review: Review):
    """
    Predict sentiment for a single text input with confidence scores.
    """
    try:
        result = predict_sentiment_advanced(review.text)
        
        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_sentiment_batch(reviews: BatchReview):
    """
    Predict sentiment for multiple texts with batch processing.
    """
    try:
        results = []
        sentiment_counts = {"pos": 0, "neg": 0}
        total_confidence = 0.0
        
        for text in reviews.texts:
            result = predict_sentiment_advanced(text)
            
            sentiment_response = SentimentResponse(
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )
            results.append(sentiment_response)
            
            # Update statistics
            sentiment_counts[result["sentiment"]] += 1
            total_confidence += result["confidence"]
        
        # Calculate summary statistics
        total_texts = len(reviews.texts)
        summary = {
            "total_texts": total_texts,
            "positive_count": sentiment_counts["pos"],
            "negative_count": sentiment_counts["neg"],
            "positive_percentage": round((sentiment_counts["pos"] / total_texts) * 100, 2),
            "negative_percentage": round((sentiment_counts["neg"] / total_texts) * 100, 2),
            "average_confidence": round(total_confidence / total_texts, 4)
        }
        
        return BatchSentimentResponse(
            results=results,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded models and their performance metrics.
    """
    try:
        model_type = "Ensemble (Naive Bayes + Logistic Regression + Random Forest)"
        features = [
            "TF-IDF Vectorization with n-grams (1-3)",
            "Advanced text preprocessing",
            "Ensemble voting (soft voting)",
            "Cross-validation scoring"
        ]
        
        training_accuracy = None
        cv_score = None
        
        if model_metrics:
            training_accuracy = model_metrics.get("accuracy")
            cv_score = model_metrics.get("cv_mean")
        
        return ModelInfo(
            model_type=model_type,
            features=features,
            training_accuracy=training_accuracy,
            cross_validation_score=cv_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_models(review: Review):
    """
    Compare predictions between advanced and legacy models (if both available).
    """
    try:
        results = {}
        
        # Advanced model prediction
        if advanced_model and advanced_vectorizer:
            processed_text = advanced_preprocess_text(review.text)
            features = advanced_vectorizer.transform([processed_text])
            prediction = advanced_model.predict(features)[0]
            probabilities = advanced_model.predict_proba(features)[0]
            
            results["advanced"] = {
                "sentiment": prediction,
                "confidence": float(max(probabilities)),
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                }
            }
        
        # Legacy model prediction
        if legacy_model and legacy_vectorizer:
            processed_text = review.text.lower().translate(str.maketrans('', '', string.punctuation))
            features = legacy_vectorizer.transform([processed_text])
            prediction = legacy_model.predict(features)[0]
            probabilities = legacy_model.predict_proba(features)[0]
            
            results["legacy"] = {
                "sentiment": prediction,
                "confidence": float(max(probabilities)),
                "probabilities": {
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                }
            }
        
        if not results:
            raise HTTPException(status_code=500, detail="No models available for comparison")
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "advanced_model_loaded": advanced_model is not None,
        "legacy_model_loaded": legacy_model is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""FastAPI service exposing the trained sentiment pipeline.

Run with:  uvicorn app:app --reload
The model artifact is produced by train.py; override its location with the
MODEL_PATH environment variable (used by the test suite).
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sentiment import SentimentModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", BASE_DIR / "artifacts" / "sentiment_pipeline.joblib")
)
METRICS_PATH = Path(os.environ.get("METRICS_PATH", BASE_DIR / "artifacts" / "metrics.json"))

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python train.py` first."
        )
    app.state.model = SentimentModel.load(MODEL_PATH)
    app.state.metrics = (
        json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    )
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="IMDB-trained sentiment classifier with exact, per-token explanations.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class TokenWeight(BaseModel):
    token: str
    weight: float = Field(
        ..., description="Exact logit contribution; positive pushes towards 'positive'."
    )


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    top_features: list[TokenWeight]


class BatchResponse(BaseModel):
    results: list[PredictionResponse]
    summary: dict


def _predict(text: str) -> PredictionResponse:
    prediction = app.state.model.predict(text)
    return PredictionResponse(
        sentiment=prediction.sentiment,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        top_features=[
            TokenWeight(token=c.token, weight=c.weight) for c in prediction.top_features
        ],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(review: ReviewRequest):
    if not review.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    return _predict(review.text)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    results = [_predict(text) for text in batch.texts]
    positives = sum(1 for r in results if r.sentiment == "positive")
    return BatchResponse(
        results=results,
        summary={
            "total": len(results),
            "positive": positives,
            "negative": len(results) - positives,
            "average_confidence": round(
                sum(r.confidence for r in results) / len(results), 4
            ),
        },
    )


@app.get("/model/info")
def model_info():
    if app.state.metrics is None:
        raise HTTPException(status_code=404, detail="No metrics file found.")
    return app.state.metrics


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}

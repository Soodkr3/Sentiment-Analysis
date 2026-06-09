from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .preprocess import clean_text


@dataclass
class TokenContribution:
    """A token's exact contribution to the prediction's logit."""

    token: str
    weight: float


@dataclass
class Prediction:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    top_features: list[TokenContribution]


class SentimentModel:
    """Wraps a fitted Pipeline of TfidfVectorizer -> LogisticRegression.

    Because the classifier is linear, every prediction is exactly
    explainable: the logit is the sum over present tokens of
    (tf-idf weight x learned coefficient) plus the intercept. A positive
    token weight pushes the prediction towards the positive class.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.vectorizer = pipeline.named_steps["tfidf"]
        self.classifier = pipeline.named_steps["classifier"]
        self.classes: list[str] = [str(c) for c in self.classifier.classes_]

    @classmethod
    def load(cls, path: str | Path) -> "SentimentModel":
        return cls(joblib.load(path))

    def predict(self, text: str, top_k: int = 8) -> Prediction:
        features = self.vectorizer.transform([clean_text(text)])
        probabilities = self.classifier.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return Prediction(
            sentiment=self.classes[best],
            confidence=float(probabilities[best]),
            probabilities={
                cls: float(p)
                for cls, p in zip(self.classes, probabilities, strict=True)
            },
            top_features=self._explain(features, top_k),
        )

    def _explain(self, features, top_k: int) -> list[TokenContribution]:
        # Coefficients refer to the logit of classes_[1] (the positive class).
        coefficients = self.classifier.coef_[0]
        present = features.nonzero()[1]
        if present.size == 0:
            return []
        vocabulary = self.vectorizer.get_feature_names_out()
        contributions = [
            TokenContribution(
                token=str(vocabulary[i]),
                weight=float(features[0, i] * coefficients[i]),
            )
            for i in present
        ]
        contributions.sort(key=lambda c: abs(c.weight), reverse=True)
        return contributions[:top_k]

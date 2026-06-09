"""Train and evaluate the sentiment pipeline on the IMDB 50k review dataset.

Usage:
    python train.py [--sample N]

Downloads the dataset on first run (cached under data/), compares a
Multinomial Naive Bayes baseline against a tuned logistic regression on a
stratified held-out test set, then refits the winner and saves:

    artifacts/sentiment_pipeline.joblib   the fitted vectorizer + classifier
    artifacts/metrics.json                everything reported below

All reported numbers come from the 20% test split the model never saw.
"""

import argparse
import json
import logging
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment.preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "imdb_dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Mirror of the standard IMDB 50k dataset (Maas et al., 2011).
DATASET_URL = (
    "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/"
    "master/IMDB-Dataset.csv"
)


def load_dataset(sample: int | None = None) -> pd.DataFrame:
    if not DATA_PATH.exists():
        log.info("Downloading IMDB dataset to %s ...", DATA_PATH)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    if {"review", "sentiment"} - set(df.columns):
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
    df = df.dropna().drop_duplicates(subset="review")
    df["review"] = df["review"].map(clean_text)
    if sample:
        df = df.groupby("sentiment").sample(n=sample // 2, random_state=42)
    log.info(
        "Loaded %d reviews (%s)", len(df), df["sentiment"].value_counts().to_dict()
    )
    return df


def build_vectorizer() -> TfidfVectorizer:
    # No stop-word removal: words like "not" carry sentiment, and bigrams
    # let the model learn negations such as "not good".
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    positive_index = list(pipeline.classes_).index("positive")
    scores = pipeline.predict_proba(X_test)[:, positive_index]
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, pos_label="positive"), 4),
        "recall": round(recall_score(y_test, predictions, pos_label="positive"), 4),
        "f1": round(f1_score(y_test, predictions, pos_label="positive"), 4),
        "roc_auc": round(roc_auc_score(y_test == "positive", scores), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["negative", "positive"]
        ).tolist(),
    }
    log.info("%s test metrics: %s", name, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Train on a balanced subsample of N reviews (for quick runs).",
    )
    args = parser.parse_args()

    df = load_dataset(args.sample)
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    log.info("Training Naive Bayes baseline...")
    baseline = Pipeline(
        [("tfidf", build_vectorizer()), ("classifier", MultinomialNB())]
    )
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate("MultinomialNB baseline", baseline, X_test, y_test)

    log.info("Tuning logistic regression (grid over C, 3-fold CV on train split)...")
    model = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    search = GridSearchCV(
        model,
        param_grid={"classifier__C": [0.25, 1.0, 4.0]},
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    log.info("Best params: %s", search.best_params_)
    model_metrics = evaluate("LogisticRegression", best, X_test, y_test)

    log.info("5-fold cross-validation of the chosen configuration on the train split...")
    cv_scores = cross_val_score(
        search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    log.info("CV accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best, PIPELINE_PATH, compress=3)
    log.info("Pipeline saved to %s", PIPELINE_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            {
                "model": "TF-IDF (1-2 grams) + LogisticRegression",
                "best_params": search.best_params_,
                "dataset": {
                    "name": "IMDB 50k movie reviews (Maas et al., 2011)",
                    "source": DATASET_URL,
                    "n_total": len(df),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                },
                "test_metrics": model_metrics,
                "baseline_naive_bayes": baseline_metrics,
                "cv_accuracy_mean": round(cv_scores.mean(), 4),
                "cv_accuracy_std": round(cv_scores.std(), 4),
                "trained_at": datetime.now(UTC).isoformat(),
                "sklearn_version": sklearn.__version__,
            },
            indent=2,
        )
    )
    log.info("Metrics saved to %s", METRICS_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Training failed")
        sys.exit(1)

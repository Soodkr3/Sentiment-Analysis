import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

POSITIVE = [
    "a wonderful brilliant film with excellent acting",
    "i loved this excellent movie it was wonderful",
    "brilliant story and excellent direction loved it",
    "wonderful cast excellent script a joy to watch",
    "loved the brilliant cinematography truly wonderful",
]
NEGATIVE = [
    "a terrible boring film with awful acting",
    "i hated this awful movie it was terrible",
    "boring story and awful direction hated it",
    "terrible cast awful script a chore to watch",
    "hated the boring pacing truly terrible",
]


@pytest.fixture(scope="session")
def tiny_pipeline_path(tmp_path_factory):
    """A small but real pipeline so tests exercise the same code paths as
    the production artifact without depending on it."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(POSITIVE + NEGATIVE, ["positive"] * 5 + ["negative"] * 5)
    path = tmp_path_factory.mktemp("artifacts") / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture(scope="session")
def metrics_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("artifacts") / "metrics.json"
    path.write_text(json.dumps({"model": "test", "test_metrics": {"accuracy": 1.0}}))
    return path

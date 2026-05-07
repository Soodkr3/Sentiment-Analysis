"""
Happy-path + error-case tests for every FastAPI endpoint in enhanced_app.py.
The `app_client` fixture (conftest.py) wires up a TestClient with mocked models.
"""

import pytest

# ---------------------------------------------------------------------------
# Root / Health
# ---------------------------------------------------------------------------


def test_root(app_client):
    r = app_client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert "version" in body
    assert "features" in body


def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "advanced_model_loaded" in body
    assert "legacy_model_loaded" in body


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


def test_predict_positive_text(app_client):
    r = app_client.post("/predict", json={"text": "This movie was absolutely amazing!"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] == "pos"
    assert 0.0 <= body["confidence"] <= 1.0
    probs = body["probabilities"]
    assert "positive" in probs
    assert "negative" in probs
    assert abs(probs["positive"] + probs["negative"] - 1.0) < 1e-6


def test_predict_missing_text_field(app_client):
    """Request body without `text` key must be rejected with 422."""
    r = app_client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_empty_string_rejected(app_client):
    """The Review model enforces min_length=1 for text."""
    r = app_client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_text_too_long_rejected(app_client):
    """Text exceeding max_length=5000 must be rejected with 422."""
    r = app_client.post("/predict", json={"text": "x" * 5001})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------


def test_predict_batch_happy_path(app_client):
    texts = ["Great movie!", "Terrible film.", "I loved it."]
    r = app_client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    summary = body["summary"]
    assert summary["total_texts"] == 3
    assert summary["positive_count"] + summary["negative_count"] == 3
    assert "average_confidence" in summary


def test_predict_batch_empty_list(app_client):
    """Empty texts list must return zero-filled summary without crashing."""
    r = app_client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 200
    body = r.json()
    summary = body["summary"]
    assert summary["total_texts"] == 0
    assert summary["positive_count"] == 0
    assert summary["negative_count"] == 0
    assert summary["average_confidence"] == 0.0


def test_predict_batch_too_many_items(app_client):
    """Sending more than 100 texts must be rejected with 422."""
    r = app_client.post("/predict/batch", json={"texts": ["text"] * 101})
    # Pydantic v1 max_items / Pydantic v2 max_length both produce 422
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------


def test_model_info(app_client):
    r = app_client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_type" in body
    assert isinstance(body["features"], list)
    assert len(body["features"]) > 0


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


def test_compare_both_models(app_client):
    r = app_client.post("/compare", json={"text": "This is a good film."})
    assert r.status_code == 200
    body = r.json()
    # At least one model result must be present
    assert "advanced" in body or "legacy" in body
    for key in body:
        assert "sentiment" in body[key]
        assert "confidence" in body[key]
        assert "probabilities" in body[key]


def test_compare_empty_text_rejected(app_client):
    r = app_client.post("/compare", json={"text": ""})
    assert r.status_code == 422

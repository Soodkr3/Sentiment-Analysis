"""Tests for model loading and availability."""


def test_at_least_one_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["advanced_model_loaded"] or data["legacy_model_loaded"]


def test_advanced_model_loaded(client):
    """Advanced model artifacts exist in backend/, so this should be True."""
    response = client.get("/health")
    data = response.json()
    assert data["advanced_model_loaded"] is True


def test_model_info_returns_cv_score(client):
    """model_metrics.json exists, so metrics should be populated."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["cross_validation_score"] is not None
    assert data["cross_validation_score"] > 0.9


def test_predict_works_after_load(client):
    """Verify a prediction can be made, confirming model is usable."""
    response = client.post("/predict", json={"text": "great film"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ("pos", "neg")

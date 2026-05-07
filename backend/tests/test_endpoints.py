"""FastAPI endpoint tests for enhanced_app."""


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "features" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "advanced_model_loaded" in data
    assert "legacy_model_loaded" in data


def test_predict_positive_happy_path(client):
    response = client.post("/predict", json={"text": "This movie was absolutely fantastic and I loved it!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] in ("pos", "neg")
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    probs = data["probabilities"]
    assert "positive" in probs
    assert "negative" in probs
    assert abs(probs["positive"] + probs["negative"] - 1.0) < 0.01


def test_predict_negative_sentiment(client):
    response = client.post("/predict", json={"text": "Terrible film, worst acting I have ever seen, complete waste of time"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ("pos", "neg")
    assert data["confidence"] > 0.5


def test_predict_empty_text_returns_422(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_batch_happy_path(client):
    texts = [
        "This movie was great!",
        "Terrible film, I hated it.",
        "Mediocre experience overall.",
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == len(texts)
    for result in data["results"]:
        assert result["sentiment"] in ("pos", "neg")
        assert 0.0 <= result["confidence"] <= 1.0
    summary = data["summary"]
    assert summary["total_texts"] == len(texts)
    assert summary["positive_count"] + summary["negative_count"] == len(texts)
    assert "average_confidence" in summary


def test_predict_batch_empty_list_returns_422(client):
    """Empty list would cause ZeroDivisionError without the guard we added."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_batch_invalid_type_returns_422(client):
    response = client.post("/predict/batch", json={"texts": "not a list"})
    assert response.status_code == 422


def test_model_info_happy_path(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


def test_model_info_has_metrics(client):
    response = client.get("/model/info")
    data = response.json()
    assert data["training_accuracy"] is not None
    assert data["cross_validation_score"] is not None
    assert 0.9 <= data["cross_validation_score"] <= 1.0


def test_compare_happy_path(client):
    response = client.post("/compare", json={"text": "This movie was great!"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    for model_result in data.values():
        assert "sentiment" in model_result
        assert "confidence" in model_result
        assert "probabilities" in model_result


def test_compare_empty_text_returns_422(client):
    response = client.post("/compare", json={"text": ""})
    assert response.status_code == 422

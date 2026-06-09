import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_pipeline_path, metrics_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_pipeline_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))
    import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    response = client.post("/predict", json={"text": "a wonderful excellent film"})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["top_features"]


def test_predict_rejects_blank_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch_predict_with_summary(client):
    response = client.post(
        "/predict/batch",
        json={"texts": ["wonderful excellent film", "terrible awful film"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["summary"]["total"] == 2
    assert body["summary"]["positive"] == 1
    assert body["summary"]["negative"] == 1


def test_batch_rejects_oversized_payload(client):
    response = client.post("/predict/batch", json={"texts": ["x"] * 101})
    assert response.status_code == 422


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["test_metrics"]["accuracy"] == 1.0

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure backend/ is on the path so `import enhanced_app` works when pytest
# is invoked from the repo root or from backend/tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _pos_model():
    m = MagicMock()
    m.predict.return_value = ["pos"]
    m.predict_proba.return_value = np.array([[0.2, 0.8]])
    return m


def _neg_model():
    m = MagicMock()
    m.predict.return_value = ["neg"]
    m.predict_proba.return_value = np.array([[0.8, 0.2]])
    return m


def _vectorizer():
    v = MagicMock()
    v.transform.return_value = MagicMock()
    return v


@pytest.fixture(scope="session")
def app_client():
    """
    FastAPI TestClient with mocked ML models.

    We patch load_models() to a no-op so the startup event doesn't try to
    read .joblib files from disk, then manually assign mock objects to the
    module-level globals.
    """
    from fastapi.testclient import TestClient

    import enhanced_app

    with patch.object(enhanced_app, "load_models"):
        enhanced_app.advanced_model = _pos_model()
        enhanced_app.advanced_vectorizer = _vectorizer()
        enhanced_app.legacy_model = _neg_model()
        enhanced_app.legacy_vectorizer = _vectorizer()
        enhanced_app.model_metrics = {"accuracy": 1.0, "cv_mean": 0.9667}

        with TestClient(enhanced_app.app) as client:
            yield client

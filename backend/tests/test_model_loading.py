"""
Tests for load_models() and the internal prediction helper.

The session-scoped `app_client` fixture (conftest.py) replaces
enhanced_app.load_models with a MagicMock for the duration of the session.
To test the *real* implementation we capture a reference at import time,
before any fixture has been set up.  Module imports happen during collection,
so this reference is always the genuine function.
"""

import json
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import enhanced_app

# Captured at collection time — before the session fixture patches the module attr.
_real_load_models = enhanced_app.load_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_globals() -> dict:
    return {
        "advanced_model": enhanced_app.advanced_model,
        "advanced_vectorizer": enhanced_app.advanced_vectorizer,
        "legacy_model": enhanced_app.legacy_model,
        "legacy_vectorizer": enhanced_app.legacy_vectorizer,
        "model_metrics": enhanced_app.model_metrics,
    }


def _restore_globals(saved: dict):
    for k, v in saved.items():
        setattr(enhanced_app, k, v)


# ---------------------------------------------------------------------------
# load_models() unit tests (call the real implementation)
# ---------------------------------------------------------------------------


def test_load_models_raises_when_no_model_files():
    """When no .joblib files exist, load_models must propagate an exception."""
    saved = _save_globals()
    try:
        enhanced_app.advanced_model = None
        enhanced_app.legacy_model = None

        with patch("os.path.exists", return_value=False):
            with pytest.raises(Exception):
                _real_load_models()
    finally:
        _restore_globals(saved)


def test_load_models_loads_advanced_model():
    """When advanced model files exist, advanced_model global must be set."""
    saved = _save_globals()
    mock_model = MagicMock()
    metrics = {"accuracy": 1.0, "cv_mean": 0.9667}

    def _exists(path):
        return "advanced" in path or "metrics" in path

    try:
        enhanced_app.advanced_model = None
        enhanced_app.advanced_vectorizer = None
        enhanced_app.legacy_model = None

        with patch("os.path.exists", side_effect=_exists), \
             patch("joblib.load", return_value=mock_model), \
             patch("builtins.open", mock_open(read_data=json.dumps(metrics))):
            _real_load_models()

        assert enhanced_app.advanced_model is not None
        assert enhanced_app.advanced_vectorizer is not None
        assert enhanced_app.model_metrics is not None
    finally:
        _restore_globals(saved)


def test_load_models_falls_back_to_legacy():
    """When only legacy model files exist, legacy_model global must be set."""
    saved = _save_globals()
    mock_model = MagicMock()

    def _exists(path):
        return "sentiment_nb_model" in path or "sentiment_vectorizer" in path

    try:
        enhanced_app.advanced_model = None
        enhanced_app.legacy_model = None

        with patch("os.path.exists", side_effect=_exists), \
             patch("joblib.load", return_value=mock_model):
            _real_load_models()

        assert enhanced_app.legacy_model is not None
    finally:
        _restore_globals(saved)


# ---------------------------------------------------------------------------
# predict_sentiment_advanced fallback path
# ---------------------------------------------------------------------------


def test_predict_uses_legacy_when_advanced_unavailable(app_client):
    """predict_sentiment_advanced must fall back to legacy if advanced is None."""
    saved = _save_globals()

    neg_model = MagicMock()
    neg_model.predict.return_value = ["neg"]
    neg_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    neg_vec = MagicMock()
    neg_vec.transform.return_value = MagicMock()

    try:
        enhanced_app.advanced_model = None
        enhanced_app.advanced_vectorizer = None
        enhanced_app.legacy_model = neg_model
        enhanced_app.legacy_vectorizer = neg_vec

        result = enhanced_app.predict_sentiment_advanced("This is terrible.")
        assert result["sentiment"] == "neg"
    finally:
        _restore_globals(saved)

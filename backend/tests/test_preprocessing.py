"""Tests for the text preprocessing pipeline."""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from enhanced_app import advanced_preprocess_text  # noqa: E402


def test_lowercase_conversion():
    assert advanced_preprocess_text("HELLO WORLD") == "hello world"


def test_punctuation_removal():
    result = advanced_preprocess_text("Hello, World! How are you?")
    assert "," not in result
    assert "!" not in result
    assert "?" not in result


def test_whitespace_normalization():
    assert advanced_preprocess_text("  hello   world  ") == "hello world"


def test_full_pipeline():
    result = advanced_preprocess_text("  This IS a Test! With PUNCTUATION...  ")
    assert result == "this is a test with punctuation"


def test_empty_string():
    assert advanced_preprocess_text("") == ""


def test_none_input():
    assert advanced_preprocess_text(None) == ""


def test_non_string_input():
    assert advanced_preprocess_text(42) == ""


def test_preserves_content_words():
    result = advanced_preprocess_text("The movie was great")
    assert "movie" in result
    assert "great" in result


def test_removes_all_punctuation_types():
    result = advanced_preprocess_text("test: result; value (check) [brackets] {braces}")
    for ch in ":;()[]{}":
        assert ch not in result

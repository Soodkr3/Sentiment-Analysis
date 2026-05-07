"""
Unit tests for the advanced_preprocess_text() function.
No ML models are needed here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from enhanced_app import advanced_preprocess_text


def test_lowercases_text():
    assert advanced_preprocess_text("HELLO WORLD") == "hello world"


def test_removes_punctuation():
    result = advanced_preprocess_text("Hello, world!")
    assert "," not in result
    assert "!" not in result
    assert "hello" in result
    assert "world" in result


def test_collapses_whitespace():
    result = advanced_preprocess_text("too   many    spaces")
    assert "  " not in result
    assert result == "too many spaces"


def test_strips_leading_trailing_whitespace():
    result = advanced_preprocess_text("   trim me   ")
    assert result == "trim me"


def test_empty_string_returns_empty():
    assert advanced_preprocess_text("") == ""


def test_non_string_returns_empty():
    assert advanced_preprocess_text(None) == ""  # type: ignore[arg-type]
    assert advanced_preprocess_text(42) == ""  # type: ignore[arg-type]


def test_combined_preprocessing():
    text = "  This MOVIE was GREAT!!! Loved it...  "
    result = advanced_preprocess_text(text)
    assert result == "this movie was great loved it"


def test_preserves_words_without_punctuation():
    result = advanced_preprocess_text("simple text")
    assert result == "simple text"


def test_removes_all_punctuation_characters():
    import string
    punctuation_text = string.punctuation
    result = advanced_preprocess_text(punctuation_text)
    assert result == ""

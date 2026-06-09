from sentiment.preprocess import clean_text


def test_strips_html_tags():
    assert clean_text("great<br /><br />movie") == "great movie"


def test_collapses_whitespace():
    assert clean_text("  too   many\n\nspaces ") == "too many spaces"


def test_non_string_input_returns_empty():
    assert clean_text(None) == ""
    assert clean_text(42) == ""


def test_preserves_case_and_punctuation():
    # Lowercasing/tokenisation belong to the vectorizer, not preprocessing.
    assert clean_text("Not good!") == "Not good!"

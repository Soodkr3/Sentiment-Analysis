import re

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace.

    Lowercasing and tokenisation are deliberately left to the TF-IDF
    vectorizer inside the pipeline, so the preprocessing applied at serving
    time can never drift from what the model was trained on.
    """
    if not isinstance(text, str):
        return ""
    text = _TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()

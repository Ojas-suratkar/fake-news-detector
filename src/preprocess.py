import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA = re.compile(r"[^a-z\s]")

def clean_text(text: str) -> str:
    """Minimal, fast text cleaner:
    - strip URLs and HTML
    - lowercase
    - keep only letters and space
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
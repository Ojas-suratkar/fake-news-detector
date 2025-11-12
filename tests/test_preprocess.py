from src.preprocess import clean_text

def test_clean_text_basic():
    t = "Hello!!! Visit https://example.com NOW <b>please</b>"
    out = clean_text(t)
    assert "http" not in out
    assert "<b>" not in out
    assert out.islower()
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("Paste any headline or short news text. Model outputs 0=REAL, 1=FAKE.")

@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).resolve().parents[1] / "models" / "fake_news_pipeline.joblib"
    if not model_path.exists():
        st.warning("Model not found. Train it first with: `python src/train.py`")
        st.stop()
    return joblib.load(model_path)

pipe = load_pipeline()

txt = st.text_area("News text:", height=150, placeholder="e.g., Breaking: Scientists discover chocolate-powered cars...")
if st.button("Check"):
    if txt.strip():
        pred = int(pipe.predict([txt])[0])
        label = "✅ REAL (0)" if pred == 0 else "🚨 FAKE (1)"
        st.subheader(label)
    else:
        st.info("Please enter some text.")
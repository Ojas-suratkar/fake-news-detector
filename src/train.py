import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

from .config import Config
from .preprocess import clean_text
from .utils import ensure_dir, save_json

def build_pipeline(max_features: int):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            preprocessor=clean_text
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    return pipe

def main(data_path: str, out_dir: str):
    cfg = Config()
    out_dir = ensure_dir(out_dir)
    df = pd.read_csv(data_path)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=cfg.test_size, random_state=cfg.random_state, stratify=df["label"]
    )

    pipe = build_pipeline(cfg.max_features)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    model_path = Path(out_dir) / cfg.model_filename
    joblib.dump(pipe, model_path)
    save_json({"accuracy": acc, "report": report}, Path(out_dir) / "metrics.json")

    print(f"Saved model to: {model_path}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "sample_fake_news.csv"))
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "models"))
    args = ap.parse_args()
    main(args.data_path, args.out_dir)
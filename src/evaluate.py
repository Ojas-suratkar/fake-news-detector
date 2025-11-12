import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    if not {"text", "label"} <= set(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")

    pipe = joblib.load(model_path)
    y_true = df["label"]
    y_pred = pipe.predict(df["text"])

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "sample_fake_news.csv"))
    ap.add_argument("--model_path", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "fake_news_pipeline.joblib"))
    args = ap.parse_args()
    main(args.data_path, args.model_path)
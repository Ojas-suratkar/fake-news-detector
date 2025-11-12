import argparse
import joblib
import json
from pathlib import Path

def main(model_path: str, text: str | None, file_path: str | None):
    pipe = joblib.load(model_path)
    outputs = []

    if text:
        pred = int(pipe.predict([text])[0])
        proba = getattr(pipe.named_steps.get("clf", None), "predict_proba", None)
        conf = None
        if proba:
            tf = pipe.named_steps["tfidf"].transform([text])
            conf = float(proba(tf)[:, pred][0])
        outputs.append({"text": text, "prediction": pred, "confidence": conf})

    if file_path:
        lines = [l.strip() for l in Path(file_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        preds = pipe.predict(lines)
        outputs.extend([{"text": t, "prediction": int(p)} for t, p in zip(lines, preds)])

    print(json.dumps(outputs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(Path(__file__).resolve().parents[1] / "models" / "fake_news_pipeline.joblib"))
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--file_path", type=str, default=None, help="Path to a .txt file with one example per line")
    args = ap.parse_args()
    main(args.model_path, args.text, args.file_path)
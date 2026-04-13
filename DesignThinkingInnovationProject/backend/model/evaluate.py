import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import DATA_DIR, MODEL_PATH
from model.features import clean_text


def main():
    data_path = DATA_DIR / "sample_reviews.csv"
    df = pd.read_csv(data_path)
    df["clean_text"] = df["review_text"].astype(str).apply(clean_text)

    X = df[["clean_text", "rating"]]
    y = df["label"].astype(str)

    pipeline = joblib.load(MODEL_PATH)
    preds = pipeline.predict(X)

    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds))


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import MODEL_PATH
from model.features import clean_text


def predict_review(review_text: str, rating: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")

    pipeline = joblib.load(MODEL_PATH)
    clean = clean_text(review_text)
    df = pd.DataFrame([{"clean_text": clean, "rating": rating}])

    label = pipeline.predict(df)[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df)[0]

    # Trust score: higher means more likely genuine.
    trust_score = None
    if proba is not None:
        classes = list(pipeline.classes_)
        if "genuine" in classes:
            trust_score = float(proba[classes.index("genuine")])
        else:
            # fallback for label naming in small datasets
            trust_score = float(max(proba))

    return label, trust_score

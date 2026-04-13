import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import DATA_DIR, MODEL_PATH, VECTORIZER_PATH
from model.features import clean_text, extra_feature_transformer


def main():
    data_path = DATA_DIR / "sample_reviews.csv"
    df = pd.read_csv(data_path)

    # Basic cleanup; kept intentionally light for explainability.
    df["clean_text"] = df["review_text"].astype(str).apply(clean_text)

    X = df[["clean_text", "rating"]]
    y = df["label"].astype(str)

    # I tried SVM at first, but Logistic Regression was faster and simpler to explain.
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_vectorizer, "clean_text"),
            ("extra", FunctionTransformer(extra_feature_transformer, validate=False), ["clean_text", "rating"]),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {score:.3f}")

    # Save both pipeline and vectorizer for transparency in report.
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(text_vectorizer, VECTORIZER_PATH)
    print("Model saved to", MODEL_PATH)


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, DEEP_MODEL_PATH, FUSION_MODEL_PATH, MODEL_METADATA_PATH, MODEL_VERSION
from data_preprocessing import build_feature_bundle, bundle_to_feature_row, prepare_training_frame

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    DEEP_STACK_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    AutoModel = None
    DEEP_STACK_AVAILABLE = False


NUMERIC_FEATURE_COLUMNS = [
    "rating",
    "token_count",
    "sentiment_score",
    "sentiment_gap",
    "readability_score",
    "reviewer_velocity",
    "repetition_ratio",
    "lexical_diversity",
    "marketing_pressure",
    "punctuation_pressure",
    "verified_purchase_flag",
    "account_age_days",
    "review_count_last_24h",
]


@dataclass
class ModelTrainingSummary:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    runtime_mode: str


if DEEP_STACK_AVAILABLE:
    class HybridReviewNet(nn.Module):
        """
        This is the intended TrustCheck 2.0 architecture for deeper training runs.
        It fuses:
        1. Transformer semantics
        2. BiLSTM sequential cues
        3. Tabular reviewer and review metadata
        """

        def __init__(
            self,
            transformer_name: str = "distilroberta-base",
            sequence_embedding_dim: int = 128,
            lstm_hidden_size: int = 96,
            tabular_dim: int = len(NUMERIC_FEATURE_COLUMNS),
            vocab_size: int = 50000,
        ):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(transformer_name)
            self.token_embedding = nn.Embedding(vocab_size, sequence_embedding_dim, padding_idx=0)
            self.sequence_encoder = nn.LSTM(
                input_size=sequence_embedding_dim,
                hidden_size=lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            transformer_width = self.transformer.config.hidden_size
            fusion_width = transformer_width + (lstm_hidden_size * 2) + 32
            self.classifier = nn.Sequential(
                nn.Linear(fusion_width, 192),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(192, 2),
            )

        def forward(self, input_ids, attention_mask, sequence_ids, tabular_features):
            transformer_output = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls_embedding = transformer_output.last_hidden_state[:, 0, :]

            sequence_embeddings = self.token_embedding(sequence_ids)
            _, (hidden_state, _) = self.sequence_encoder(sequence_embeddings)
            lstm_embedding = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)

            tabular_embedding = self.tabular_encoder(tabular_features)
            fused_representation = torch.cat(
                (cls_embedding, lstm_embedding, tabular_embedding),
                dim=1,
            )
            return self.classifier(fused_representation)


def train_fusion_baseline(dataset_path: Path | None = None) -> ModelTrainingSummary:
    dataset_path = dataset_path or (DATA_DIR / "sample_reviews.csv")
    review_data = pd.read_csv(dataset_path)
    prepared_data = prepare_training_frame(review_data)

    X = prepared_data[["clean_text", *NUMERIC_FEATURE_COLUMNS]]
    y = prepared_data["label"].replace({"fake": "suspicious", "genuine": "genuine"})

    text_transform = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=4000,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    numeric_transform = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    feature_pipeline = ColumnTransformer(
        transformers=[
            ("text", text_transform, "clean_text"),
            ("numeric", numeric_transform, NUMERIC_FEATURE_COLUMNS),
        ]
    )

    trust_engine = LogisticRegression(
        max_iter=1400,
        class_weight="balanced",
        solver="liblinear",
    )
    training_pipeline = Pipeline(
        steps=[
            ("feature_pipeline", feature_pipeline),
            ("trust_engine", trust_engine),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    training_pipeline.fit(X_train, y_train)

    predictions = training_pipeline.predict(X_test)
    summary = ModelTrainingSummary(
        accuracy=round(accuracy_score(y_test, predictions), 3),
        precision=round(precision_score(y_test, predictions, pos_label="suspicious"), 3),
        recall=round(recall_score(y_test, predictions, pos_label="suspicious"), 3),
        f1_score=round(f1_score(y_test, predictions, pos_label="suspicious"), 3),
        runtime_mode="fusion-fallback",
    )

    joblib.dump(training_pipeline, FUSION_MODEL_PATH)
    MODEL_METADATA_PATH.write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "runtime_mode": summary.runtime_mode,
                "metrics": summary.__dict__,
                "deep_stack_available": DEEP_STACK_AVAILABLE,
                "deep_checkpoint_present": DEEP_MODEL_PATH.exists(),
                "dataset_path": str(dataset_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


class TrustCheckInferenceEngine:
    def __init__(self):
        self._fusion_pipeline = None
        self._metadata = None

    def _ensure_runtime_model(self):
        if not FUSION_MODEL_PATH.exists():
            train_fusion_baseline()

        if self._fusion_pipeline is None:
            self._fusion_pipeline = joblib.load(FUSION_MODEL_PATH)

        if self._metadata is None:
            if MODEL_METADATA_PATH.exists():
                self._metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
            else:
                self._metadata = {
                    "model_version": MODEL_VERSION,
                    "runtime_mode": "fusion-fallback",
                }

    def analyze_payload(self, payload) -> dict:
        self._ensure_runtime_model()

        feature_bundle = build_feature_bundle(
            review_text=f"{payload.title} {payload.text}".strip(),
            rating=payload.rating,
            verified_purchase=payload.verified_purchase,
            account_age_days=payload.account_age_days,
            review_count_last_24h=payload.review_count_last_24h,
        )
        feature_row = bundle_to_feature_row(feature_bundle, payload.rating)
        input_frame = pd.DataFrame([feature_row])[["clean_text", *NUMERIC_FEATURE_COLUMNS]]

        predicted_label = self._fusion_pipeline.predict(input_frame)[0]
        probability_map = {}
        if hasattr(self._fusion_pipeline, "predict_proba"):
            class_names = list(self._fusion_pipeline.classes_)
            class_scores = self._fusion_pipeline.predict_proba(input_frame)[0]
            probability_map = dict(zip(class_names, [float(score) for score in class_scores]))

        suspicious_probability = probability_map.get("suspicious", 0.5 if predicted_label == "suspicious" else 0.25)
        trust_score = round(1.0 - suspicious_probability, 3)
        anomaly_score = round(suspicious_probability, 3)

        key_signals = self._compose_key_signals(payload, feature_bundle, anomaly_score)
        derived_features = feature_row | {
            "predicted_label": predicted_label,
            "trust_score": trust_score,
            "anomaly_score": anomaly_score,
        }

        runtime_mode = "deep-hybrid-ready" if DEEP_STACK_AVAILABLE else self._metadata.get("runtime_mode", "fusion-fallback")
        if DEEP_MODEL_PATH.exists():
            runtime_mode = "deep-hybrid-checkpoint"

        return {
            "label": predicted_label,
            "trust_score": trust_score,
            "anomaly_score": anomaly_score,
            "runtime_mode": runtime_mode,
            "model_version": self._metadata.get("model_version", MODEL_VERSION),
            "key_signals": key_signals,
            "derived_features": derived_features,
        }

    def _compose_key_signals(self, payload, bundle, anomaly_score: float) -> list[str]:
        notes: list[str] = []

        if bundle.sentiment_gap >= 0.8:
            notes.append("The review sentiment and the star rating do not line up cleanly.")
        if bundle.reviewer_velocity >= 1.5:
            notes.append("Reviewer velocity is high, which can hint at bursty activity.")
        if bundle.marketing_pressure >= 0.08:
            notes.append("Promotional wording is stronger than what we usually see in organic reviews.")
        if bundle.repetition_ratio >= 0.2:
            notes.append("The text repeats words more than normal, which can signal templated language.")
        if not payload.verified_purchase:
            notes.append("The review is marked as non-verified, so metadata trust is lower.")
        if payload.account_age_days <= 14:
            notes.append("The account is quite new, so the reviewer profile is still low-confidence.")
        if anomaly_score <= 0.4:
            notes.append("Text semantics and metadata look reasonably consistent overall.")

        return notes[:4]

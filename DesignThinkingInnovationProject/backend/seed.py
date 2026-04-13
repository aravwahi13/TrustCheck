import csv
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import DATA_DIR, MODEL_VERSION
from data_preprocessing import build_feature_bundle
from db import get_db


def init_db():
    schema_path = BACKEND_DIR / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with get_db() as conn:
        conn.executescript(schema_sql)


def seed_reviews():
    sample_path = DATA_DIR / "sample_reviews.csv"
    if not sample_path.exists():
        print("Sample dataset not found:", sample_path)
        return

    with get_db() as conn, open(sample_path, "r", encoding="utf-8") as sample_file:
        reader = csv.DictReader(sample_file)
        for row in reader:
            rating = int(row["rating"])
            verified_purchase = str(row.get("verified_purchase", "0")).strip() in {"1", "true", "True"}
            account_age_days = int(row.get("account_age_days", 0))
            review_count_last_24h = int(row.get("review_count_last_24h", 0))
            label = "suspicious" if row["label"].strip().lower() == "fake" else "genuine"

            combined_text = f"{row.get('title', '')} {row.get('text', '')}".strip()
            bundle = build_feature_bundle(
                review_text=combined_text,
                rating=rating,
                verified_purchase=verified_purchase,
                account_age_days=account_age_days,
                review_count_last_24h=review_count_last_24h,
            )
            trust_score = 0.84 if label == "genuine" else 0.18
            anomaly_score = round(1.0 - trust_score, 3)

            review_cursor = conn.execute(
                """
                INSERT INTO amazon_reviews (
                    rating, title, text, images, asin, parent_asin, user_id, timestamp, verified_purchase, helpful_vote
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rating,
                    row["title"],
                    row["text"],
                    json.dumps([]),
                    row["asin"],
                    row["parent_asin"],
                    row["user_id"],
                    int(row["timestamp"]),
                    int(verified_purchase),
                    int(row.get("helpful_vote", 0)),
                ),
            )
            review_id = int(review_cursor.lastrowid)

            conn.execute(
                """
                INSERT INTO review_analysis (
                    review_id,
                    product_url,
                    account_age_days,
                    review_count_last_24h,
                    sentiment_score,
                    sentiment_gap,
                    readability_score,
                    reviewer_velocity,
                    repetition_ratio,
                    lexical_diversity,
                    marketing_pressure,
                    punctuation_pressure,
                    predicted_label,
                    trust_score,
                    anomaly_score,
                    runtime_mode,
                    model_version,
                    source_mode
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    row.get("product_url", ""),
                    account_age_days,
                    review_count_last_24h,
                    bundle.sentiment_score,
                    bundle.sentiment_gap,
                    bundle.readability_score,
                    bundle.reviewer_velocity,
                    bundle.repetition_ratio,
                    bundle.lexical_diversity,
                    bundle.marketing_pressure,
                    bundle.punctuation_pressure,
                    label,
                    trust_score,
                    anomaly_score,
                    "seed-baseline",
                    MODEL_VERSION,
                    "demo-seed",
                ),
            )


def _schema_matches_expected() -> bool:
    with get_db() as conn:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('amazon_reviews', 'review_analysis')"
        ).fetchall()
        if {row["name"] for row in table_rows} != {"amazon_reviews", "review_analysis"}:
            return False

        raw_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(amazon_reviews)").fetchall()
        }
        analysis_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(review_analysis)").fetchall()
        }

        return {
            "rating", "title", "text", "images", "asin", "parent_asin",
            "user_id", "timestamp", "verified_purchase", "helpful_vote",
        }.issubset(raw_columns) and {
            "review_id", "product_url", "account_age_days", "review_count_last_24h",
            "sentiment_score", "sentiment_gap", "predicted_label", "trust_score",
            "anomaly_score", "runtime_mode", "model_version", "source_mode",
        }.issubset(analysis_columns)


def bootstrap_demo_state():
    if not _schema_matches_expected():
        init_db()
        seed_reviews()
        return

    with get_db() as conn:
        existing_count = conn.execute("SELECT COUNT(*) AS count FROM amazon_reviews").fetchone()["count"]
        if existing_count == 0:
            seed_reviews()


if __name__ == "__main__":
    init_db()
    seed_reviews()
    print("Database initialized with Amazon-style review records and TrustCheck analysis rows.")

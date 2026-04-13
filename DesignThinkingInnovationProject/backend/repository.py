import json

from db import get_db


def insert_review_analysis(payload: dict, analysis: dict, source_mode: str) -> int:
    with get_db() as conn:
        review_cursor = conn.execute(
            """
            INSERT INTO amazon_reviews (
                rating,
                title,
                text,
                images,
                asin,
                parent_asin,
                user_id,
                timestamp,
                verified_purchase,
                helpful_vote
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("rating"),
                payload.get("title", ""),
                payload.get("text", ""),
                json.dumps(payload.get("images", [])),
                payload.get("asin"),
                payload.get("parent_asin"),
                payload.get("user_id"),
                payload.get("timestamp"),
                int(bool(payload.get("verified_purchase"))),
                payload.get("helpful_vote", 0),
            ),
        )
        review_id = int(review_cursor.lastrowid)

        analysis_cursor = conn.execute(
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
                payload.get("product_url"),
                payload.get("account_age_days", 0),
                payload.get("review_count_last_24h", 0),
                analysis["derived_features"]["sentiment_score"],
                analysis["derived_features"]["sentiment_gap"],
                analysis["derived_features"]["readability_score"],
                analysis["derived_features"]["reviewer_velocity"],
                analysis["derived_features"]["repetition_ratio"],
                analysis["derived_features"]["lexical_diversity"],
                analysis["derived_features"]["marketing_pressure"],
                analysis["derived_features"]["punctuation_pressure"],
                analysis["label"],
                analysis["trust_score"],
                analysis["anomaly_score"],
                analysis["runtime_mode"],
                analysis["model_version"],
                source_mode,
            ),
        )
        return int(analysis_cursor.lastrowid)


def fetch_recent_reviews(limit: int = 8):
    with get_db() as conn:
        return conn.execute(
            """
            SELECT
                ra.id,
                ar.asin,
                ar.parent_asin,
                ar.user_id,
                ar.rating,
                ar.title,
                ar.text,
                ar.verified_purchase,
                ar.helpful_vote,
                ra.predicted_label,
                ra.trust_score,
                ra.anomaly_score,
                ra.runtime_mode,
                ra.source_mode,
                ra.created_at
            FROM review_analysis ra
            JOIN amazon_reviews ar ON ar.id = ra.review_id
            ORDER BY ra.created_at DESC, ra.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def fetch_history(limit: int = 100):
    with get_db() as conn:
        return conn.execute(
            """
            SELECT
                ra.id,
                ar.asin,
                ar.parent_asin,
                ar.user_id,
                ar.rating,
                ar.title,
                ar.text,
                ar.verified_purchase,
                ar.helpful_vote,
                ar.timestamp,
                ra.account_age_days,
                ra.review_count_last_24h,
                ra.reviewer_velocity,
                ra.predicted_label,
                ra.trust_score,
                ra.runtime_mode,
                ra.created_at
            FROM review_analysis ra
            JOIN amazon_reviews ar ON ar.id = ra.review_id
            ORDER BY ra.created_at DESC, ra.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def fetch_dashboard_metrics():
    with get_db() as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total_reviews,
                SUM(CASE WHEN ra.predicted_label = 'suspicious' THEN 1 ELSE 0 END) AS suspicious_reviews,
                ROUND(AVG(ra.trust_score), 3) AS average_trust_score,
                ROUND(AVG(CASE WHEN ar.verified_purchase = 1 THEN 1.0 ELSE 0.0 END), 3) AS verified_purchase_share
            FROM review_analysis ra
            JOIN amazon_reviews ar ON ar.id = ra.review_id
            """
        ).fetchone()

        risky_rows = conn.execute(
            """
            SELECT
                ar.asin,
                ar.user_id,
                ar.rating,
                ar.text,
                ra.trust_score,
                ra.predicted_label,
                ra.created_at
            FROM review_analysis ra
            JOIN amazon_reviews ar ON ar.id = ra.review_id
            ORDER BY ra.anomaly_score DESC, ra.id DESC
            LIMIT 6
            """
        ).fetchall()

    return summary, risky_rows

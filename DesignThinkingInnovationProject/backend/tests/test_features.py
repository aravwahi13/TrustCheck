from data_preprocessing import (
    build_feature_bundle,
    clean_review_text,
    compute_flesch_kincaid_grade,
    compute_sentiment_gap,
)


def test_clean_review_text_removes_urls_and_normalizes_case():
    cleaned = clean_review_text("Great product!!! Visit https://example.com NOW")
    assert "http" not in cleaned
    assert cleaned.startswith("great product")


def test_sentiment_gap_detects_rating_text_mismatch():
    gap = compute_sentiment_gap(rating=5, sentiment_score=-0.4)
    assert gap > 1.0


def test_readability_score_is_numeric():
    score = compute_flesch_kincaid_grade("The setup was easy. The battery lasted for six hours.")
    assert isinstance(score, float)


def test_feature_bundle_includes_velocity_and_metadata_flags():
    bundle = build_feature_bundle(
        review_text="Amazing purchase, works well, everyone should buy this now!",
        rating=5,
        verified_purchase=False,
        account_age_days=5,
        review_count_last_24h=18,
    )
    assert bundle.reviewer_velocity > 0.7
    assert bundle.verified_purchase_flag == 0

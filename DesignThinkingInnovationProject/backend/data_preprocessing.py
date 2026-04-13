import math
import re
from dataclasses import asdict, dataclass

import pandas as pd


WORD_RE = re.compile(r"[a-zA-Z']+")
SENTENCE_RE = re.compile(r"[.!?]+")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

POSITIVE_WORDS = {
    "amazing", "awesome", "best", "brilliant", "comfortable", "excellent",
    "fantastic", "fast", "good", "great", "love", "perfect", "premium",
    "reliable", "solid", "useful", "worth", "smooth", "happy", "recommend",
}
NEGATIVE_WORDS = {
    "awful", "bad", "broken", "cheap", "confusing", "delay", "disappointed",
    "fake", "hate", "issue", "poor", "refund", "scam", "slow", "terrible",
    "unhappy", "useless", "weak", "worst", "problem",
}
MARKETING_WORDS = {
    "buy", "guaranteed", "must", "perfect", "premium", "unbelievable",
    "elite", "luxury", "flawless", "incredible", "everyone", "today",
}


@dataclass
class FeatureBundle:
    clean_text: str
    token_count: int
    sentiment_score: float
    sentiment_gap: float
    readability_score: float
    reviewer_velocity: float
    repetition_ratio: float
    lexical_diversity: float
    marketing_pressure: float
    punctuation_pressure: float
    verified_purchase_flag: int
    account_age_days: int
    review_count_last_24h: int


def clean_review_text(text: str) -> str:
    normalized = URL_RE.sub(" ", text or "")
    normalized = normalized.replace("\n", " ").replace("\r", " ")
    normalized = re.sub(r"[^a-zA-Z0-9!?'\s]", " ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip().lower()
    return normalized


def tokenize_review(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def estimate_syllables(word: str) -> int:
    word = word.lower()
    if not word:
        return 1

    vowel_runs = re.findall(r"[aeiouy]+", word)
    syllable_count = len(vowel_runs)
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1
    return max(1, syllable_count)


def compute_sentiment_score(text: str) -> float:
    tokens = tokenize_review(text)
    if not tokens:
        return 0.0

    positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    raw_score = (positive_hits - negative_hits) / max(len(tokens), 1)
    return max(-1.0, min(1.0, raw_score * 5))


def expected_sentiment_from_rating(rating: int) -> float:
    return (rating - 3) / 2


def compute_sentiment_gap(rating: int, sentiment_score: float) -> float:
    return abs(expected_sentiment_from_rating(rating) - sentiment_score)


def compute_flesch_kincaid_grade(text: str) -> float:
    clean_text = text.strip()
    if not clean_text:
        return 0.0

    words = tokenize_review(clean_text)
    sentence_count = len(SENTENCE_RE.findall(clean_text)) or 1
    word_count = len(words) or 1
    syllable_count = sum(estimate_syllables(word) for word in words)

    grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
    return round(grade, 3)


def compute_repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0

    duplicate_count = len(tokens) - len(set(tokens))
    return round(duplicate_count / len(tokens), 3)


def compute_lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 3)


def compute_marketing_pressure(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    marketing_hits = sum(1 for token in tokens if token in MARKETING_WORDS)
    return round(marketing_hits / len(tokens), 3)


def compute_punctuation_pressure(text: str) -> float:
    if not text:
        return 0.0
    emphasis_count = text.count("!") + text.count("?")
    uppercase_letters = sum(1 for char in text if char.isupper())
    normalized = (emphasis_count * 0.4) + (uppercase_letters / max(1, len(text)))
    return round(normalized, 3)


def compute_reviewer_velocity(review_count_last_24h: int) -> float:
    return round(review_count_last_24h / 24.0, 3)


def build_feature_bundle(
    review_text: str,
    rating: int,
    verified_purchase: bool,
    account_age_days: int,
    review_count_last_24h: int,
) -> FeatureBundle:
    clean_text = clean_review_text(review_text)
    tokens = tokenize_review(clean_text)
    sentiment_score = compute_sentiment_score(clean_text)

    return FeatureBundle(
        clean_text=clean_text,
        token_count=len(tokens),
        sentiment_score=round(sentiment_score, 3),
        sentiment_gap=round(compute_sentiment_gap(rating, sentiment_score), 3),
        readability_score=compute_flesch_kincaid_grade(review_text),
        reviewer_velocity=compute_reviewer_velocity(review_count_last_24h),
        repetition_ratio=compute_repetition_ratio(tokens),
        lexical_diversity=compute_lexical_diversity(tokens),
        marketing_pressure=compute_marketing_pressure(tokens),
        punctuation_pressure=compute_punctuation_pressure(review_text),
        verified_purchase_flag=1 if verified_purchase else 0,
        account_age_days=account_age_days,
        review_count_last_24h=review_count_last_24h,
    )


def bundle_to_feature_row(bundle: FeatureBundle, rating: int) -> dict[str, float | int | str]:
    row = asdict(bundle)
    row["rating"] = rating
    return row


def prepare_training_frame(review_data: pd.DataFrame) -> pd.DataFrame:
    prepared_rows: list[dict[str, float | int | str]] = []

    for row in review_data.to_dict(orient="records"):
        combined_text = f"{row.get('title', '')} {row.get('text', '')}".strip()
        bundle = build_feature_bundle(
            review_text=combined_text,
            rating=int(row["rating"]),
            verified_purchase=bool(row.get("verified_purchase", False)),
            account_age_days=int(row.get("account_age_days", 0)),
            review_count_last_24h=int(row.get("review_count_last_24h", 0)),
        )
        feature_row = bundle_to_feature_row(bundle, int(row["rating"]))
        feature_row.update(
            {
                "asin": str(row.get("asin", "B0DEMO0001")),
                "parent_asin": str(row.get("parent_asin", "PARENT-DEMO-001")),
                "product_url": str(row.get("product_url", "")),
                "user_id": str(row.get("user_id", "guest-user")),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "label": str(row["label"]),
            }
        )
        prepared_rows.append(feature_row)

    prepared_frame = pd.DataFrame(prepared_rows)
    # Using grade magnitude directly can overreact on tiny texts, so a log transform calms it down.
    prepared_frame["readability_score"] = prepared_frame["readability_score"].apply(
        lambda score: round(math.copysign(math.log1p(abs(score)), score), 3)
    )
    return prepared_frame

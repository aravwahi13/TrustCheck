DROP TABLE IF EXISTS review_analysis;
DROP TABLE IF EXISTS amazon_reviews;

CREATE TABLE amazon_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating REAL NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    images TEXT NOT NULL DEFAULT '[]',
    asin TEXT NOT NULL,
    parent_asin TEXT NOT NULL,
    user_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    verified_purchase INTEGER NOT NULL DEFAULT 0,
    helpful_vote INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE review_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL,
    product_url TEXT,
    account_age_days INTEGER DEFAULT 0,
    review_count_last_24h INTEGER DEFAULT 0,
    sentiment_score REAL,
    sentiment_gap REAL,
    readability_score REAL,
    reviewer_velocity REAL,
    repetition_ratio REAL,
    lexical_diversity REAL,
    marketing_pressure REAL,
    punctuation_pressure REAL,
    predicted_label TEXT,
    trust_score REAL,
    anomaly_score REAL,
    runtime_mode TEXT,
    model_version TEXT,
    source_mode TEXT DEFAULT 'manual',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES amazon_reviews (id)
);

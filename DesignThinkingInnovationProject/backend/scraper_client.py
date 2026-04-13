import os
import re

import httpx
import pandas as pd

from config import DATA_DIR


ASIN_RE = re.compile(r"(B0[A-Z0-9]{8})", re.IGNORECASE)


class ReviewScraperClient:
    def __init__(self):
        self.api_url = os.getenv("SCRAPER_API_URL")
        self.api_key = os.getenv("SCRAPER_API_KEY")

    def fetch_reviews(self, product_url: str, max_reviews: int = 5) -> dict:
        if self.api_url and self.api_key:
            try:
                return self._fetch_live_reviews(product_url, max_reviews)
            except Exception as exc:
                return {
                    "source_mode": "live-api-fallback",
                    "message": f"Live scraping failed, so TrustCheck switched to demo mode: {exc}",
                    "reviews": self._build_demo_reviews(product_url, max_reviews),
                }

        return {
            "source_mode": "demo-catalog",
            "message": "Live scraping is not configured, so demo review samples were used for this URL.",
            "reviews": self._build_demo_reviews(product_url, max_reviews),
        }

    def _fetch_live_reviews(self, product_url: str, max_reviews: int) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.get(
            self.api_url,
            params={"url": product_url, "limit": max_reviews},
            headers=headers,
            timeout=25.0,
        )
        response.raise_for_status()
        payload = response.json()
        raw_reviews = payload.get("reviews") or payload.get("data", {}).get("reviews") or []

        normalized = []
        for index, item in enumerate(raw_reviews[:max_reviews], start=1):
            normalized.append(
                {
                    "rating": int(float(item.get("rating", 3))),
                    "title": item.get("title") or "",
                    "text": item.get("text") or item.get("review_text") or "",
                    "images": item.get("images") or [],
                    "asin": item.get("asin") or f"B0LIVE{index:05d}",
                    "parent_asin": item.get("parent_asin") or item.get("asin") or f"PARENT-LIVE-{index}",
                    "user_id": item.get("user_id") or item.get("author") or f"live-user-{index}",
                    "timestamp": int(item.get("timestamp", 1712707200)),
                    "verified_purchase": bool(item.get("verified_purchase", False)),
                    "helpful_vote": int(item.get("helpful_vote", 0)),
                    "product_url": product_url,
                    "account_age_days": int(item.get("account_age_days", 0)),
                    "review_count_last_24h": int(item.get("review_count_last_24h", 1)),
                }
            )

        return {
            "source_mode": "live-api",
            "message": "Reviews were fetched from the configured scraper integration.",
            "reviews": normalized,
        }

    def _build_demo_reviews(self, product_url: str, max_reviews: int) -> list[dict]:
        review_catalog = pd.read_csv(DATA_DIR / "sample_reviews.csv")
        asin_match = ASIN_RE.search(product_url)
        scoped_catalog = pd.DataFrame()

        if asin_match:
            scoped_catalog = review_catalog[
                review_catalog["asin"].str.upper() == asin_match.group(1).upper()
            ]
            if not scoped_catalog.empty:
                review_catalog = scoped_catalog

        if len(review_catalog) < max_reviews and not scoped_catalog.empty:
            remaining_catalog = pd.read_csv(DATA_DIR / "sample_reviews.csv")
            remaining_catalog = remaining_catalog[
                remaining_catalog["asin"].str.upper() != asin_match.group(1).upper()
            ]
            review_catalog = pd.concat([review_catalog, remaining_catalog], ignore_index=True)

        demo_reviews = []
        for row in review_catalog.head(max_reviews).to_dict(orient="records"):
            demo_reviews.append(
                {
                    "rating": int(row["rating"]),
                    "title": row["title"],
                    "text": row["text"],
                    "images": [],
                    "asin": row["asin"],
                    "parent_asin": row["parent_asin"],
                    "user_id": row["user_id"],
                    "timestamp": int(row["timestamp"]),
                    "verified_purchase": bool(row.get("verified_purchase", False)),
                    "helpful_vote": int(row.get("helpful_vote", 0)),
                    "product_url": product_url,
                    "account_age_days": int(row.get("account_age_days", 0)),
                    "review_count_last_24h": int(row.get("review_count_last_24h", 1)),
                }
            )
        return demo_reviews

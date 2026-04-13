from typing import Literal

from pydantic import BaseModel, Field


class ReviewPayload(BaseModel):
    rating: int = Field(ge=1, le=5)
    title: str = Field(default="")
    text: str = Field(min_length=5, max_length=4000)
    images: list[str] = Field(default_factory=list)
    asin: str = Field(default="B0DEMO0001")
    parent_asin: str = Field(default="PARENT-DEMO-001")
    user_id: str = Field(default="guest-user")
    timestamp: int = Field(default=1712707200, ge=0)
    verified_purchase: bool = False
    helpful_vote: int = Field(default=0, ge=0)
    product_url: str | None = Field(default=None)
    account_age_days: int = Field(default=0, ge=0)
    review_count_last_24h: int = Field(default=0, ge=0)


class UrlReviewRequest(BaseModel):
    product_url: str = Field(min_length=8)
    max_reviews: int = Field(default=5, ge=1, le=20)


class ReviewAnalysisResponse(BaseModel):
    label: Literal["genuine", "suspicious"]
    trust_score: float
    anomaly_score: float
    runtime_mode: str
    model_version: str
    key_signals: list[str]
    derived_features: dict[str, float | int | str]


class UrlAnalysisResponse(BaseModel):
    product_url: str
    source_mode: str
    fetched_count: int
    suspicious_count: int
    average_trust_score: float
    items: list[ReviewAnalysisResponse]

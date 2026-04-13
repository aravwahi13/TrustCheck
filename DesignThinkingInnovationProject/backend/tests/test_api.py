from fastapi.testclient import TestClient

from app import app
from seed import bootstrap_demo_state


bootstrap_demo_state()
client = TestClient(app)


def test_home_page_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "TrustCheck 2.0" in response.text


def test_api_analyze_returns_hybrid_fields():
    payload = {
        "asin": "B0TEST0001",
        "parent_asin": "PARENT-TEST-001",
        "user_id": "reviewer-alpha",
        "rating": 5,
        "title": "Good first impression",
        "text": "Great value and the packaging was neat, but this sounds too polished to trust blindly.",
        "verified_purchase": True,
        "helpful_vote": 4,
        "timestamp": 1712707200,
        "account_age_days": 120,
        "review_count_last_24h": 1,
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "trust_score" in body
    assert "derived_features" in body
    assert "runtime_mode" in body


def test_api_analyze_url_supports_demo_scraper():
    response = client.post(
        "/api/analyze-url",
        json={"product_url": "https://www.amazon.com/dp/B0TCHECK01", "max_reviews": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fetched_count"] == 3
    assert "source_mode" in body


def test_csv_upload_flow_accepts_valid_template_shape():
    csv_content = (
        "rating,title,text,images,asin,parent_asin,user_id,timestamp,verified_purchase,helpful_vote,product_url,account_age_days,review_count_last_24h\n"
        "5,Useful kettle,Heats nicely,[],B0CSV00001,PARENT-CSV-001,AUPLOAD001,1712707200,1,5,https://www.amazon.com/dp/B0CSV00001,200,1\n"
    )
    response = client.post(
        "/upload-csv",
        files={"review_csv": ("reviews.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    assert "CSV upload processed successfully" in response.text

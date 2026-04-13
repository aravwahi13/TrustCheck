import sys
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import APP_HOST, APP_PORT, MODEL_VERSION, STATIC_DIR, TEMPLATE_DIR
from model_engine import TrustCheckInferenceEngine
from repository import fetch_dashboard_metrics, fetch_history, fetch_recent_reviews, insert_review_analysis
from schemas import ReviewPayload, UrlReviewRequest
from scraper_client import ReviewScraperClient
from seed import bootstrap_demo_state


@asynccontextmanager
async def lifespan(_: FastAPI):
    bootstrap_demo_state()
    yield


app = FastAPI(title="TrustCheck 2.0", version=MODEL_VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

trust_engine = TrustCheckInferenceEngine()
review_scraper = ReviewScraperClient()

UPLOAD_REQUIRED_COLUMNS = [
    "rating",
    "title",
    "text",
    "images",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
    "product_url",
    "account_age_days",
    "review_count_last_24h",
]


def _build_manual_payload(
    asin: str,
    parent_asin: str,
    user_id: str,
    rating: int,
    title: str,
    text: str,
    helpful_vote: int,
    timestamp: int,
    product_url: str | None,
    verified_purchase: str | None,
    account_age_days: int,
    review_count_last_24h: int,
) -> ReviewPayload:
    return ReviewPayload(
        asin=asin.strip() or "B0DEMO0001",
        parent_asin=parent_asin.strip() or "PARENT-DEMO-001",
        user_id=user_id.strip() or "guest-user",
        rating=rating,
        title=title.strip(),
        text=text.strip(),
        helpful_vote=helpful_vote,
        timestamp=timestamp,
        product_url=(product_url or "").strip() or None,
        verified_purchase=verified_purchase is not None,
        account_age_days=account_age_days,
        review_count_last_24h=review_count_last_24h,
    )


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _build_payload_from_row(row: dict[str, str]) -> ReviewPayload:
    images_raw = (row.get("images") or "").strip()
    if images_raw and images_raw != "[]":
        image_items = [item.strip() for item in images_raw.split("|") if item.strip()]
    else:
        image_items = []

    return ReviewPayload(
        rating=int(float(row["rating"])),
        title=(row.get("title") or "").strip(),
        text=(row.get("text") or "").strip(),
        images=image_items,
        asin=(row.get("asin") or "").strip(),
        parent_asin=(row.get("parent_asin") or "").strip(),
        user_id=(row.get("user_id") or "").strip(),
        timestamp=int(float(row.get("timestamp") or 0)),
        verified_purchase=_parse_bool(row.get("verified_purchase", "0")),
        helpful_vote=int(float(row.get("helpful_vote") or 0)),
        product_url=(row.get("product_url") or "").strip() or None,
        account_age_days=int(float(row.get("account_age_days") or 0)),
        review_count_last_24h=int(float(row.get("review_count_last_24h") or 0)),
    )


def _process_payloads(payloads: list[ReviewPayload], source_mode: str):
    batch_results = []
    suspicious_count = 0
    trust_total = 0.0

    for payload in payloads:
        analysis = trust_engine.analyze_payload(payload)
        insert_review_analysis(payload.model_dump(), analysis, source_mode=source_mode)
        batch_results.append({"payload": payload.model_dump(), "analysis": analysis})
        suspicious_count += 1 if analysis["label"] == "suspicious" else 0
        trust_total += analysis["trust_score"]

    average_trust_score = round(trust_total / max(len(batch_results), 1), 3)
    return batch_results, suspicious_count, average_trust_score


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    recent_reviews = fetch_recent_reviews(limit=8)
    dashboard_summary, _ = fetch_dashboard_metrics()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "recent_reviews": recent_reviews,
            "dashboard_summary": dashboard_summary,
            "page_title": "TrustCheck 2.0",
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    dashboard_summary, risky_rows = fetch_dashboard_metrics()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": dashboard_summary,
            "risky_rows": risky_rows,
            "page_title": "Dashboard",
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    history_rows = fetch_history(limit=100)
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "history_rows": history_rows,
            "page_title": "User History",
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
def analyze_review(
    request: Request,
    asin: str = Form("B0TCHECK01"),
    parent_asin: str = Form("PARENT-TCHECK-01"),
    user_id: str = Form("guest-user"),
    rating: int = Form(...),
    title: str = Form(""),
    text: str = Form(...),
    helpful_vote: int = Form(0),
    timestamp: int = Form(1712707200),
    product_url: str = Form(""),
    verified_purchase: str | None = Form(default=None),
    account_age_days: int = Form(0),
    review_count_last_24h: int = Form(0),
):
    payload = _build_manual_payload(
        asin=asin,
        parent_asin=parent_asin,
        user_id=user_id,
        rating=rating,
        title=title,
        text=text,
        helpful_vote=helpful_vote,
        timestamp=timestamp,
        product_url=product_url,
        verified_purchase=verified_purchase,
        account_age_days=account_age_days,
        review_count_last_24h=review_count_last_24h,
    )
    analysis = trust_engine.analyze_payload(payload)
    record_id = insert_review_analysis(payload.model_dump(), analysis, source_mode="manual-form")

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "analysis": analysis,
            "payload": payload.model_dump(),
            "record_id": record_id,
            "page_title": "Analysis Result",
        },
    )


@app.post("/analyze-url", response_class=HTMLResponse)
def analyze_product_url(
    request: Request,
    product_url: str = Form(...),
    max_reviews: int = Form(5),
):
    url_request = UrlReviewRequest(product_url=product_url, max_reviews=max_reviews)
    fetched_bundle = review_scraper.fetch_reviews(url_request.product_url, url_request.max_reviews)
    payloads = [ReviewPayload(**item) for item in fetched_bundle["reviews"]]
    batch_results, suspicious_count, average_trust_score = _process_payloads(
        payloads,
        source_mode=fetched_bundle["source_mode"],
    )
    return templates.TemplateResponse(
        request,
        "batch_result.html",
        {
            "product_url": url_request.product_url,
            "batch_results": batch_results,
            "source_mode": fetched_bundle["source_mode"],
            "source_message": fetched_bundle["message"],
            "fetched_count": len(batch_results),
            "suspicious_count": suspicious_count,
            "average_trust_score": average_trust_score,
            "page_title": "URL Analysis",
        },
    )


@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv_reviews(
    request: Request,
    review_csv: UploadFile = File(...),
):
    csv_bytes = await review_csv.read()
    csv_text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(csv_text))

    missing_columns = [column for column in UPLOAD_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
    if missing_columns:
        return templates.TemplateResponse(
            request,
            "batch_result.html",
            {
                "product_url": "CSV Upload",
                "batch_results": [],
                "source_mode": "csv-upload-error",
                "source_message": "Upload failed. Missing required columns: " + ", ".join(missing_columns),
                "fetched_count": 0,
                "suspicious_count": 0,
                "average_trust_score": 0,
                "page_title": "CSV Upload",
            },
            status_code=400,
        )

    payloads = []
    for row_number, row in enumerate(reader, start=2):
        if not any((value or "").strip() for value in row.values()):
            continue
        try:
            payloads.append(_build_payload_from_row(row))
        except Exception as exc:
            return templates.TemplateResponse(
                request,
                "batch_result.html",
                {
                    "product_url": "CSV Upload",
                    "batch_results": [],
                    "source_mode": "csv-upload-error",
                    "source_message": f"Upload failed on row {row_number}: {exc}",
                    "fetched_count": 0,
                    "suspicious_count": 0,
                    "average_trust_score": 0,
                    "page_title": "CSV Upload",
                },
                status_code=400,
            )

    batch_results, suspicious_count, average_trust_score = _process_payloads(
        payloads,
        source_mode="csv-upload",
    )
    return templates.TemplateResponse(
        request,
        "batch_result.html",
        {
            "product_url": review_csv.filename or "Uploaded CSV",
            "batch_results": batch_results,
            "source_mode": "csv-upload",
            "source_message": "CSV upload processed successfully. Required columns are Amazon-style plus TrustCheck metadata fields.",
            "fetched_count": len(batch_results),
            "suspicious_count": suspicious_count,
            "average_trust_score": average_trust_score,
            "page_title": "CSV Upload",
        },
    )


@app.post("/api/analyze", response_class=JSONResponse)
def api_analyze(payload: ReviewPayload):
    analysis = trust_engine.analyze_payload(payload)
    insert_review_analysis(payload.model_dump(), analysis, source_mode="api")
    return analysis


@app.post("/api/analyze-url", response_class=JSONResponse)
def api_analyze_url(url_request: UrlReviewRequest):
    fetched_bundle = review_scraper.fetch_reviews(url_request.product_url, url_request.max_reviews)
    payloads = [ReviewPayload(**item) for item in fetched_bundle["reviews"]]
    batch_results, suspicious_count, average_trust_score = _process_payloads(
        payloads,
        source_mode=fetched_bundle["source_mode"],
    )
    items = [item["analysis"] for item in batch_results]

    return {
        "product_url": url_request.product_url,
        "source_mode": fetched_bundle["source_mode"],
        "message": fetched_bundle["message"],
        "fetched_count": len(items),
        "suspicious_count": suspicious_count,
        "average_trust_score": average_trust_score,
        "items": items,
    }


@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok", "service": "TrustCheck 2.0"}


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

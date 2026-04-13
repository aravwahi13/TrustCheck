## TrustCheck – Hybrid Fake Review Detection System

This repository contains the full project deliverables for the CSET210 DTI Milestone 1 and the end-to-end prototype implementation.

**Chosen domain:** Machine Learning / Natural Language Processing  
**Tech stack:** Python, FastAPI, scikit-learn, SQLite, HTML/CSS/JS


### Quick start
**Python version note:** scikit-learn does not yet support Python 3.14.  
Please use Python 3.11 or 3.12 for this project.

1. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```
pip install -r backend/requirements.txt
```
Optional for the heavier deep-learning stack:
```
pip install -r backend/requirements-deep-learning.txt
```
3. Initialize the local database and seed example data:
```
python backend/seed.py
```
4. Train the runnable fusion model:
```
python backend/train_trustcheck2.py
```
5. Run the app:
```
python backend/app.py
```
6. Open in browser:
```
http://127.0.0.1:8000
```

### CSV upload format
TrustCheck 2.0 supports bulk upload from the home page.

Required CSV columns:
```
rating,title,text,images,asin,parent_asin,user_id,timestamp,verified_purchase,helpful_vote,product_url,account_age_days,review_count_last_24h
```

Notes:
- `images` can be `[]` or pipe-separated URLs.
- `timestamp` should be a Unix timestamp in seconds.
- `verified_purchase` accepts `1/0`, `true/false`, or `yes/no`.
- `account_age_days` and `review_count_last_24h` are TrustCheck metadata features used in the scoring model.

A ready-to-use sample is available at:
`backend/static/amazon_upload_template.csv`

### Project structure
```
backend/
  app.py
  config.py
  db.py
  schema.sql
  seed.py
  data_preprocessing.py
  model_engine.py
  scraper_client.py
  repository.py
  model/
  data/
  templates/
  static/
  tests/
report/
presentation/
```

### Notes
- The dataset in `backend/data/sample_reviews.csv` is still intentionally small so the app stays runnable during evaluation.
- For a larger experiment, swap in Amazon Reviews 2023 / SNAP-scale data and train the deep model path with the optional requirements.

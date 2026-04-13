from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
ARTIFACT_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

DB_PATH = BASE_DIR / "trustcheck.db"

# Legacy baseline artifact paths are kept so the original milestone prototype still exists.
LEGACY_MODEL_PATH = MODEL_DIR / "trustcheck_model.joblib"
LEGACY_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

# TrustCheck 2.0 artifacts.
FUSION_MODEL_PATH = ARTIFACT_DIR / "trustcheck_fusion.joblib"
MODEL_METADATA_PATH = ARTIFACT_DIR / "trustcheck_metadata.json"
DEEP_MODEL_PATH = ARTIFACT_DIR / "trustcheck_hybrid_deep.pt"

MODEL_VERSION = "trustcheck-2.0"
APP_HOST = "127.0.0.1"
APP_PORT = 8000

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

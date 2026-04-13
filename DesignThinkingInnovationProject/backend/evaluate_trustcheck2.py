import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import MODEL_METADATA_PATH


if __name__ == "__main__":
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("No TrustCheck 2.0 metadata found. Train the fusion baseline first.")

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    print(json.dumps(metadata, indent=2))

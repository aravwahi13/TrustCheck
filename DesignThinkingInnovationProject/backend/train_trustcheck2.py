import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from model_engine import train_fusion_baseline


if __name__ == "__main__":
    summary = train_fusion_baseline()
    print("Fusion baseline trained successfully.")
    print(summary)

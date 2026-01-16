import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

PRED_LOG_FILE = LOG_DIR / "predictions.jsonl"

def log_prediction(payload: dict):
    """
    Writes one JSON per line (JSONL format).
    Perfect for production logging + later analysis.
    """
    event = {
        "ts_utc": datetime.utcnow().isoformat(),
        **payload
    }
    with open(PRED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

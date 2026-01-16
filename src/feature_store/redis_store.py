import json
import redis
from typing import List, Dict, Any

REDIS_HOST = "localhost"
REDIS_PORT = 6379

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def vehicle_key(vehicle_id: str) -> str:
    return f"telemetry:{vehicle_id}"

def ingest_telemetry(vehicle_id: str, record: Dict[str, Any], keep_last: int = 60):
    """
    Store telemetry record in Redis list, keep only last N.
    """
    key = vehicle_key(vehicle_id)
    r.lpush(key, json.dumps(record))
    r.ltrim(key, 0, keep_last - 1)

def fetch_last_n(vehicle_id: str, n: int = 60) -> List[Dict[str, Any]]:
    """
    Fetch last N telemetry records (oldest -> newest).
    """
    key = vehicle_key(vehicle_id)
    raw = r.lrange(key, 0, n - 1)
    # Redis list is newest-first due to LPUSH
    return [json.loads(x) for x in reversed(raw)]

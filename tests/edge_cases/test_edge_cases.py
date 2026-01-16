from fastapi.testclient import TestClient
from src.api.main import APP

client = TestClient(APP)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_missing_required_features():
    payload = {
        "vehicle_id": "V001",
        "timestamp": "2025-11-01 08:30:00",
        "features": {
            # missing differential_pressure_kpa etc.
            "engine_load_pct": 40,
            "engine_rpm": 1500
        }
    }
    r = client.post("/predict/soot-load", json=payload)
    assert r.status_code == 422

def test_out_of_range_engine_load():
    payload = {
        "vehicle_id": "V001",
        "timestamp": "2025-11-01 08:30:00",
        "features": {
            "vehicle_speed_kmh": 50,
            "engine_load_pct": 500,  # invalid
            "engine_rpm": 1500,
            "ambient_temp_c": 25,
            "exhaust_temp_pre_dpf_c": 300,
            "differential_pressure_kpa": 10
        }
    }
    r = client.post("/predict/soot-load", json=payload)
    assert r.status_code == 422


"""
FINAL API TEST SCRIPT - Matches Your Exact Schema
Run with: python test_api_final.py (while API server is running)
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

def test_endpoint(name, method, url, payload=None):
    """Test a single endpoint and return result"""
    print(f"\n{'─'*70}")
    print(f"TEST: {name}")
    print(f"{'─'*70}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            print(f"Request Payload:")
            print(json.dumps(payload, indent=2))
            response = requests.post(url, json=payload, timeout=10)
        
        print(f"\nStatus Code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response:")
            print(json.dumps(response_json, indent=2))
            
            if response.status_code == 200:
                print(f"\n✅ {name} - PASSED")
                return True, response_json
            else:
                print(f"\n❌ {name} - FAILED")
                return False, response_json
        except:
            print(f"Response Text: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"\n❌ {name} - ERROR: {str(e)}")
        return False, None

# =============================================================================
# TEST SUITE
# =============================================================================

print_section("DPF SOOT PREDICTION - FINAL API TEST SUITE")
print(f"API Base URL: {API_BASE_URL}")
print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = {}

# =============================================================================
# TEST 1: Health Check
# =============================================================================
print_section("TEST 1: Health Check")
success, data = test_endpoint(
    "GET /health",
    "GET",
    f"{API_BASE_URL}/health"
)
results["health"] = success

# =============================================================================
# TEST 2: Model Info
# =============================================================================
print_section("TEST 2: Model Info")
success, data = test_endpoint(
    "GET /model/info",
    "GET",
    f"{API_BASE_URL}/model/info"
)
results["model_info"] = success

# =============================================================================
# TEST 3: Single Prediction - /predict/soot-load
# =============================================================================
print_section("TEST 3: Single Prediction")

# Based on your PredictRequest schema:
# - vehicle_id: str
# - timestamp: str
# - features: Dict[str, Any]
predict_payload = {
    "vehicle_id": "TEST_V001",
    "timestamp": "2026-02-08 12:30:00",  # Your schema expects this format
    "features": {
        # Required features (from validate_features function)
        "engine_load_pct": 62.5,
        "engine_rpm": 1750,
        "vehicle_speed_kmh": 75.0,
        "exhaust_temp_pre_dpf_c": 390.0,
        "differential_pressure_kpa": 14.2,
        
        # Optional but commonly used features
        "ambient_temp_c": 25.0,
        "exhaust_temp_post_dpf_c": 370.0,
        "exhaust_flow_rate": 150.0
    }
}

success, data = test_endpoint(
    "POST /predict/soot-load",
    "POST",
    f"{API_BASE_URL}/predict/soot-load",
    predict_payload
)
results["predict_single"] = success

# =============================================================================
# TEST 4: Batch Prediction - /predict/batch
# =============================================================================
print_section("TEST 4: Batch Prediction")

# Based on BatchPredictRequest schema:
# - items: List[PredictRequest]
batch_payload = {
    "items": [
        {
            "vehicle_id": "BATCH_V001",
            "timestamp": "2026-02-08 09:00:00",
            "features": {
                "engine_load_pct": 60.0,
                "engine_rpm": 1700,
                "vehicle_speed_kmh": 70.0,
                "exhaust_temp_pre_dpf_c": 385.0,
                "differential_pressure_kpa": 14.0,
                "ambient_temp_c": 25.0,
                "exhaust_temp_post_dpf_c": 365.0,
                "exhaust_flow_rate": 145.0
            }
        },
        {
            "vehicle_id": "BATCH_V002",
            "timestamp": "2026-02-08 09:15:00",
            "features": {
                "engine_load_pct": 55.0,
                "engine_rpm": 1650,
                "vehicle_speed_kmh": 65.0,
                "exhaust_temp_pre_dpf_c": 400.0,
                "differential_pressure_kpa": 10.5,
                "ambient_temp_c": 23.0,
                "exhaust_temp_post_dpf_c": 380.0,
                "exhaust_flow_rate": 155.0
            }
        },
        {
            "vehicle_id": "BATCH_V003",
            "timestamp": "2026-02-08 09:30:00",
            "features": {
                "engine_load_pct": 70.0,
                "engine_rpm": 1850,
                "vehicle_speed_kmh": 85.0,
                "exhaust_temp_pre_dpf_c": 420.0,
                "differential_pressure_kpa": 8.0,
                "ambient_temp_c": 22.0,
                "exhaust_temp_post_dpf_c": 395.0,
                "exhaust_flow_rate": 165.0
            }
        }
    ]
}

success, data = test_endpoint(
    "POST /predict/batch",
    "POST",
    f"{API_BASE_URL}/predict/batch",
    batch_payload
)
results["predict_batch"] = success

# =============================================================================
# TEST 5: Ingest Telemetry - /ingest/telemetry
# =============================================================================
print_section("TEST 5: Ingest Telemetry")

# Based on IngestRequest schema:
# - vehicle_id: str
# - record: TelemetryRecord
ingest_payload = {
    "vehicle_id": "RT_V001",
    "record": {
        "timestamp": "2026-02-08 13:00:00",
        "vehicle_speed_kmh": 70.0,
        "engine_load_pct": 55.0,
        "engine_rpm": 1700.0,
        "ambient_temp_c": 25.0,
        "exhaust_temp_pre_dpf_c": 390.0,
        "exhaust_temp_post_dpf_c": 370.0,
        "exhaust_flow_rate": 150.0,
        "differential_pressure_kpa": 12.0
    }
}

success, data = test_endpoint(
    "POST /ingest/telemetry",
    "POST",
    f"{API_BASE_URL}/ingest/telemetry",
    ingest_payload
)
results["ingest_telemetry"] = success

# =============================================================================
# TEST 6: Predict from Raw Telemetry - /predict/from-raw
# =============================================================================
print_section("TEST 6: Predict from Raw Telemetry")

# First, ingest multiple records to build a window
print("Ingesting 15 telemetry records to build history...")
for i in range(15):
    ingest_payload = {
        "vehicle_id": "RT_V002",
        "record": {
            "timestamp": f"2026-02-08 10:{i:02d}:00",
            "vehicle_speed_kmh": 70.0 + i,
            "engine_load_pct": 55.0 + i * 0.5,
            "engine_rpm": 1700.0 + i * 10,
            "ambient_temp_c": 25.0,
            "exhaust_temp_pre_dpf_c": 390.0 + i * 2,
            "exhaust_temp_post_dpf_c": 370.0 + i * 2,
            "exhaust_flow_rate": 150.0 + i,
            "differential_pressure_kpa": 12.0 + i * 0.3
        }
    }
    requests.post(f"{API_BASE_URL}/ingest/telemetry", json=ingest_payload, timeout=5)

print("Done ingesting.\n")

# Based on PredictFromRawRequest schema:
# - vehicle_id: str
# - last_n: int (default 60)
predict_raw_payload = {
    "vehicle_id": "RT_V002",
    "last_n": 15
}

success, data = test_endpoint(
    "POST /predict/from-raw",
    "POST",
    f"{API_BASE_URL}/predict/from-raw",
    predict_raw_payload
)
results["predict_from_raw"] = success

# =============================================================================
# TEST 7: Edge Cases
# =============================================================================
print_section("TEST 7: Edge Cases")

# Edge Case 1: High soot load scenario
high_soot_payload = {
    "vehicle_id": "EDGE_HIGH",
    "timestamp": "2026-02-08 14:00:00",
    "features": {
        "engine_load_pct": 85.0,
        "engine_rpm": 2000,
        "vehicle_speed_kmh": 30.0,  # Low speed, high load
        "exhaust_temp_pre_dpf_c": 320.0,  # Low temp (no passive regen)
        "differential_pressure_kpa": 25.0,  # High pressure (blocked)
        "ambient_temp_c": 25.0,
        "exhaust_temp_post_dpf_c": 310.0,
        "exhaust_flow_rate": 120.0
    }
}

success, data = test_endpoint(
    "Edge Case: High Soot Load",
    "POST",
    f"{API_BASE_URL}/predict/soot-load",
    high_soot_payload
)
results["edge_high_soot"] = success
if success and data:
    print(f"Recommendation: {data.get('recommended_action')}")
    print(f"Predicted Soot: {data.get('soot_pred_pct', 0):.2f}%")

# Edge Case 2: Low soot load scenario (recent regen)
low_soot_payload = {
    "vehicle_id": "EDGE_LOW",
    "timestamp": "2026-02-08 14:15:00",
    "features": {
        "engine_load_pct": 50.0,
        "engine_rpm": 1600,
        "vehicle_speed_kmh": 90.0,  # Highway speed
        "exhaust_temp_pre_dpf_c": 450.0,  # High temp (passive regen)
        "differential_pressure_kpa": 3.0,  # Low pressure (clean)
        "ambient_temp_c": 25.0,
        "exhaust_temp_post_dpf_c": 430.0,
        "exhaust_flow_rate": 180.0
    }
}

success, data = test_endpoint(
    "Edge Case: Low Soot Load",
    "POST",
    f"{API_BASE_URL}/predict/soot-load",
    low_soot_payload
)
results["edge_low_soot"] = success
if success and data:
    print(f"Recommendation: {data.get('recommended_action')}")
    print(f"Predicted Soot: {data.get('soot_pred_pct', 0):.2f}%")

# Edge Case 3: Missing optional features (should work with defaults)
minimal_payload = {
    "vehicle_id": "EDGE_MINIMAL",
    "timestamp": "2026-02-08 14:30:00",
    "features": {
        # Only required features
        "engine_load_pct": 60.0,
        "engine_rpm": 1700,
        "vehicle_speed_kmh": 70.0,
        "exhaust_temp_pre_dpf_c": 390.0,
        "differential_pressure_kpa": 12.0
    }
}

success, data = test_endpoint(
    "Edge Case: Minimal Features",
    "POST",
    f"{API_BASE_URL}/predict/soot-load",
    minimal_payload
)
results["edge_minimal"] = success

# Edge Case 4: Out of range values (should fail validation)
invalid_payload = {
    "vehicle_id": "EDGE_INVALID",
    "timestamp": "2026-02-08 14:45:00",
    "features": {
        "engine_load_pct": 150.0,  # Out of range (0-100)
        "engine_rpm": 1700,
        "vehicle_speed_kmh": 70.0,
        "exhaust_temp_pre_dpf_c": 390.0,
        "differential_pressure_kpa": 12.0
    }
}

success, data = test_endpoint(
    "Edge Case: Invalid Range (should fail)",
    "POST",
    f"{API_BASE_URL}/predict/soot-load",
    invalid_payload
)
results["edge_invalid"] = not success  # We expect this to fail
if not success:
    print("✅ Correctly rejected out-of-range value")

# =============================================================================
# SUMMARY
# =============================================================================
print_section("TEST SUMMARY")

total = len(results)
passed = sum(1 for v in results.values() if v)
failed = total - passed

print(f"Total Tests: {total}")
print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")

if failed > 0:
    print("\n❌ Failed Tests:")
    for test_name, success in results.items():
        if not success:
            print(f"  - {test_name}")

print("\n" + "="*70)
print("Test Complete!")
print("="*70)

# Save results
with open('final_api_test_results.json', 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "results": results
    }, f, indent=2)

print("\n📄 Results saved to: final_api_test_results.json")

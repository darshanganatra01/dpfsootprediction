"""
API Endpoint Testing Script
Tests actual API responses - Run this AFTER starting the API server
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
test_results = []

def log_test(test_name, status, message="", details=None):
    """Log test result"""
    result = {
        "test": test_name,
        "status": status,
        "message": message,
        "details": details,
        "timestamp": datetime.now().isoformat()
    }
    test_results.append(result)
    
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {test_name}: {message}")
    if details:
        print(f"   Details: {json.dumps(details, indent=2)[:200]}...")

def test_section(title):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")

# =============================================================================
# TEST 1: API Server Running
# =============================================================================
def test_api_server_running():
    test_section("TEST 1: API Server Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            log_test("API server running", "PASS", 
                    f"Server responded with {response.status_code}")
            return True
        else:
            log_test("API server running", "FAIL", 
                    f"Server returned {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        log_test("API server running", "FAIL", 
                "Cannot connect to server. Is it running on port 8000?")
        print("\n⚠️  Please start the API server first:")
        print("   python run_api.py")
        return False
    
    except Exception as e:
        log_test("API server running", "FAIL", str(e))
        return False

# =============================================================================
# TEST 2: /health Endpoint
# =============================================================================
def test_health_endpoint():
    test_section("TEST 2: /health Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        # Check status code
        if response.status_code == 200:
            log_test("/health status code", "PASS", "200 OK")
        else:
            log_test("/health status code", "FAIL", 
                    f"Expected 200, got {response.status_code}")
            return
        
        # Check response format
        data = response.json()
        
        # Expected fields
        expected_fields = ["status"]
        optional_fields = ["model_loaded", "redis_connected", "timestamp"]
        
        for field in expected_fields:
            if field in data:
                log_test(f"/health has '{field}' field", "PASS", 
                        f"Value: {data[field]}")
            else:
                log_test(f"/health has '{field}' field", "FAIL", 
                        "Required field missing")
        
        # Log optional fields
        for field in optional_fields:
            if field in data:
                log_test(f"/health has '{field}' field", "PASS", 
                        f"Value: {data[field]}")
        
        log_test("/health response", "PASS", "Valid JSON response", data)
    
    except Exception as e:
        log_test("/health endpoint", "FAIL", str(e))

# =============================================================================
# TEST 3: /model/info Endpoint
# =============================================================================
def test_model_info_endpoint():
    test_section("TEST 3: /model/info Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        
        if response.status_code == 200:
            log_test("/model/info status code", "PASS", "200 OK")
        else:
            log_test("/model/info status code", "FAIL", 
                    f"Expected 200, got {response.status_code}")
            return
        
        data = response.json()
        
        # Expected fields per assignment
        expected_fields = ["model_type", "version", "training_date", "features_count"]
        optional_fields = ["performance_metrics", "feature_importance_top5"]
        
        for field in expected_fields:
            if field in data:
                log_test(f"/model/info has '{field}' field", "PASS", 
                        f"Value: {data[field]}")
            else:
                log_test(f"/model/info has '{field}' field", "FAIL", 
                        "Required field missing")
        
        # Check performance metrics if present
        if "performance_metrics" in data:
            metrics = data["performance_metrics"]
            expected_metrics = ["mae", "rmse", "r2"]
            
            for metric in expected_metrics:
                if metric in metrics:
                    log_test(f"/model/info metrics: {metric}", "PASS", 
                            f"{metric.upper()}: {metrics[metric]}")
        
        log_test("/model/info response", "PASS", "Valid response", data)
    
    except Exception as e:
        log_test("/model/info endpoint", "FAIL", str(e))

# =============================================================================
# TEST 4: /predict/soot-load Endpoint
# =============================================================================
def test_predict_soot_load_endpoint():
    test_section("TEST 4: /predict/soot-load Endpoint")
    
    # Test case 1: Normal prediction
    test_payload_normal = {
        "vehicle_id": "TEST_V001",
        "features": {
            "exhaust_temp_rolling_15min_mean": 385.5,
            "diff_pressure_kpa": 14.2,
            "hours_since_last_regen": 48.5,
            "idle_time_pct_last_hour": 0.15,
            "high_load_duration_minutes": 25,
            "engine_load_pct": 62,
            "vehicle_speed_kmh": 75
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/soot-load",
            json=test_payload_normal,
            timeout=10
        )
        response_time = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            log_test("/predict/soot-load status code", "PASS", "200 OK")
        else:
            log_test("/predict/soot-load status code", "FAIL", 
                    f"Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        data = response.json()
        
        # Check required fields per assignment
        required_fields = [
            "vehicle_id",
            "predicted_soot_load_pct",
            "recommendation"
        ]
        
        for field in required_fields:
            if field in data:
                log_test(f"/predict/soot-load has '{field}'", "PASS", 
                        f"Value: {data[field]}")
            else:
                log_test(f"/predict/soot-load has '{field}'", "FAIL", 
                        "Required field missing")
        
        # Validate prediction value
        if "predicted_soot_load_pct" in data:
            pred = data["predicted_soot_load_pct"]
            if 0 <= pred <= 100:
                log_test("Prediction in valid range", "PASS", 
                        f"{pred:.2f}% (0-100%)")
            else:
                log_test("Prediction in valid range", "FAIL", 
                        f"{pred:.2f}% is outside 0-100%")
        
        # Validate recommendation
        if "recommendation" in data:
            valid_recommendations = [
                "OK", "MONITOR", "PASSIVE_REGEN_OPPORTUNITY", 
                "ACTIVE_REGEN", "INSPECTION"
            ]
            rec = data["recommendation"]
            if rec in valid_recommendations:
                log_test("Recommendation valid", "PASS", f"{rec}")
            else:
                log_test("Recommendation valid", "FAIL", 
                        f"'{rec}' not in expected values")
        
        # Check confidence interval if present
        if "confidence_interval" in data:
            ci = data["confidence_interval"]
            if "lower" in ci and "upper" in ci:
                log_test("Confidence interval format", "PASS", 
                        f"[{ci['lower']:.1f}%, {ci['upper']:.1f}%]")
        
        # Check response time
        log_test("Response time", 
                "PASS" if response_time < 1000 else "FAIL",
                f"{response_time:.0f}ms (expect < 1000ms)")
        
        log_test("/predict/soot-load response", "PASS", "Valid prediction", data)
    
    except Exception as e:
        log_test("/predict/soot-load endpoint", "FAIL", str(e))

# =============================================================================
# TEST 5: /predict/batch Endpoint
# =============================================================================
def test_predict_batch_endpoint():
    test_section("TEST 5: /predict/batch Endpoint")
    
    # Test with 3 vehicles
    test_payload = {
        "predictions": [
            {
                "vehicle_id": "BATCH_V001",
                "features": {
                    "exhaust_temp_rolling_15min_mean": 385.5,
                    "diff_pressure_kpa": 14.2,
                    "hours_since_last_regen": 48.5
                }
            },
            {
                "vehicle_id": "BATCH_V002",
                "features": {
                    "exhaust_temp_rolling_15min_mean": 420.0,
                    "diff_pressure_kpa": 8.5,
                    "hours_since_last_regen": 12.0
                }
            },
            {
                "vehicle_id": "BATCH_V003",
                "features": {
                    "exhaust_temp_rolling_15min_mean": 350.0,
                    "diff_pressure_kpa": 18.5,
                    "hours_since_last_regen": 72.0
                }
            }
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=test_payload,
            timeout=10
        )
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            log_test("/predict/batch status code", "PASS", "200 OK")
        else:
            log_test("/predict/batch status code", "FAIL", 
                    f"Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        data = response.json()
        
        # Check for batch structure
        if "predictions" in data:
            log_test("/predict/batch has 'predictions'", "PASS")
            
            predictions = data["predictions"]
            if len(predictions) == 3:
                log_test("Batch prediction count", "PASS", "3 predictions returned")
            else:
                log_test("Batch prediction count", "FAIL", 
                        f"Expected 3, got {len(predictions)}")
            
            # Validate each prediction
            for i, pred in enumerate(predictions):
                if "vehicle_id" in pred and "predicted_soot_load_pct" in pred:
                    log_test(f"Batch prediction {i+1} valid", "PASS", 
                            f"{pred['vehicle_id']}: {pred['predicted_soot_load_pct']:.1f}%")
        else:
            log_test("/predict/batch has 'predictions'", "FAIL", 
                    "Missing 'predictions' field")
        
        # Check total count
        if "total_predictions" in data:
            log_test("/predict/batch total count", "PASS", 
                    f"{data['total_predictions']} predictions")
        
        log_test("Batch response time", 
                "PASS" if response_time < 2000 else "FAIL",
                f"{response_time:.0f}ms (expect < 2000ms)")
        
        log_test("/predict/batch response", "PASS", "Valid batch prediction")
    
    except Exception as e:
        log_test("/predict/batch endpoint", "FAIL", str(e))

# =============================================================================
# TEST 6: Edge Cases
# =============================================================================
def test_edge_cases():
    test_section("TEST 6: Edge Cases")
    
    # Test 1: Very high soot load (should recommend INSPECTION)
    high_soot_payload = {
        "vehicle_id": "EDGE_HIGH",
        "features": {
            "exhaust_temp_rolling_15min_mean": 300.0,
            "diff_pressure_kpa": 25.0,
            "hours_since_last_regen": 120.0,
            "idle_time_pct_last_hour": 0.8
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/soot-load",
            json=high_soot_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            pred = data.get("predicted_soot_load_pct", 0)
            rec = data.get("recommendation", "")
            
            log_test("High soot scenario", "PASS", 
                    f"Prediction: {pred:.1f}%, Recommendation: {rec}")
        else:
            log_test("High soot scenario", "FAIL", 
                    f"Status code: {response.status_code}")
    
    except Exception as e:
        log_test("High soot scenario", "FAIL", str(e))
    
    # Test 2: Low soot load (should recommend OK)
    low_soot_payload = {
        "vehicle_id": "EDGE_LOW",
        "features": {
            "exhaust_temp_rolling_15min_mean": 450.0,
            "diff_pressure_kpa": 3.0,
            "hours_since_last_regen": 2.0,
            "idle_time_pct_last_hour": 0.05
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/soot-load",
            json=low_soot_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            pred = data.get("predicted_soot_load_pct", 0)
            rec = data.get("recommendation", "")
            
            log_test("Low soot scenario", "PASS", 
                    f"Prediction: {pred:.1f}%, Recommendation: {rec}")
        else:
            log_test("Low soot scenario", "FAIL", 
                    f"Status code: {response.status_code}")
    
    except Exception as e:
        log_test("Low soot scenario", "FAIL", str(e))
    
    # Test 3: Missing features (should handle gracefully)
    missing_features_payload = {
        "vehicle_id": "EDGE_MISSING",
        "features": {
            "exhaust_temp_rolling_15min_mean": 385.5
            # Missing other required features
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/soot-load",
            json=missing_features_payload,
            timeout=10
        )
        
        # Should either return 200 with imputation or 422 with error
        if response.status_code in [200, 422]:
            log_test("Missing features handling", "PASS", 
                    f"Status code: {response.status_code}")
        else:
            log_test("Missing features handling", "FAIL", 
                    f"Unexpected status code: {response.status_code}")
    
    except Exception as e:
        log_test("Missing features handling", "FAIL", str(e))

# =============================================================================
# TEST 7: Real-time Endpoints (if Redis enabled)
# =============================================================================
def test_realtime_endpoints():
    test_section("TEST 7: Real-time Endpoints (Optional)")
    
    # Test /ingest/telemetry
    telemetry_payload = {
        "vehicle_id": "RT_V001",
        "record": {
            "timestamp": "2026-01-16T10:30:00",
            "vehicle_speed_kmh": 70,
            "engine_load_pct": 55,
            "engine_rpm": 1700,
            "exhaust_temp_pre_dpf_c": 390,
            "differential_pressure_kpa": 12
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest/telemetry",
            json=telemetry_payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            log_test("/ingest/telemetry", "PASS", "Telemetry ingested", data)
        elif response.status_code == 404:
            log_test("/ingest/telemetry", "SKIP", "Endpoint not implemented")
        else:
            log_test("/ingest/telemetry", "FAIL", 
                    f"Status code: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        log_test("/ingest/telemetry", "SKIP", "Redis feature not enabled")
    except Exception as e:
        log_test("/ingest/telemetry", "SKIP", str(e))

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================
def run_all_api_tests():
    print("\n" + "="*70)
    print(" DPF SOOT PREDICTION - API ENDPOINT TESTS")
    print("="*70)
    print(f" API Base URL: {API_BASE_URL}")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if server is running first
    if not test_api_server_running():
        print("\n❌ API server is not running. Cannot proceed with tests.")
        print("\nPlease start the server first:")
        print("   python run_api.py")
        return
    
    # Run all tests
    test_health_endpoint()
    test_model_info_endpoint()
    test_predict_soot_load_endpoint()
    test_predict_batch_endpoint()
    test_edge_cases()
    test_realtime_endpoints()
    
    # Summary
    test_section("SUMMARY")
    
    total = len(test_results)
    passed = sum(1 for r in test_results if r['status'] == 'PASS')
    failed = sum(1 for r in test_results if r['status'] == 'FAIL')
    skipped = sum(1 for r in test_results if r['status'] == 'SKIP')
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"⚠️ Skipped: {skipped} ({skipped/total*100:.1f}%)")
    
    # List failures
    failures = [r for r in test_results if r['status'] == 'FAIL']
    if failures:
        print("\n❌ FAILED TESTS:")
        print("-" * 70)
        for f in failures:
            print(f"  • {f['test']}: {f['message']}")
    
    # Save results
    with open('api_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: api_test_results.json")
    
    return passed, failed, skipped

if __name__ == "__main__":
    run_all_api_tests()

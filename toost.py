"""
Comprehensive Test Suite for DPF Soot Prediction System
Run this to identify issues before grading
"""

import sys
import os
import json
from pathlib import Path

# Test results collector
test_results = []

def log_test(test_name, status, message="", details=None):
    """Log test result"""
    result = {
        "test": test_name,
        "status": status,  # PASS, FAIL, SKIP
        "message": message,
        "details": details
    }
    test_results.append(result)
    
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {test_name}: {message}")
    if details:
        print(f"   Details: {details}")

def test_section(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

# =============================================================================
# TEST 1: File Structure & Dependencies
# =============================================================================
def test_file_structure():
    test_section("TEST 1: File Structure & Dependencies")
    
    required_files = [
        "requirements.txt",
        "Dockerfile",
        "run_generate.py",
        "run_join.py",
        "run_features.py",
        "run_train.py",
        "run_api.py"
    ]
    
    required_dirs = [
        "src/data_generation",
        "src/data_engineering",
        "src/features",
        "src/models",
        "src/api",
        "src/recommendation",
        "tests"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            log_test(f"File exists: {file}", "PASS")
        else:
            log_test(f"File exists: {file}", "FAIL", f"Missing file: {file}")
    
    for dir in required_dirs:
        if os.path.exists(dir):
            log_test(f"Directory exists: {dir}", "PASS")
        else:
            log_test(f"Directory exists: {dir}", "FAIL", f"Missing directory: {dir}")

# =============================================================================
# TEST 2: Data Generation
# =============================================================================
def test_data_generation():
    test_section("TEST 2: Data Generation")
    
    try:
        import pandas as pd
        
        # Check if data files exist
        data_files = {
            "telemetry": "data/telemetry.parquet",
            "trips": "data/trips.parquet",
            "maintenance": "data/maintenance.parquet"
        }
        
        for name, filepath in data_files.items():
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                
                # Check data size
                row_count = len(df)
                col_count = len(df.columns)
                
                log_test(
                    f"Data file: {name}",
                    "PASS",
                    f"{row_count} rows, {col_count} columns"
                )
                
                # Check for required columns based on assignment
                if name == "telemetry":
                    required_cols = ["vehicle_id", "timestamp", "engine_load_pct"]
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        log_test(f"{name} columns", "FAIL", f"Missing: {missing}")
                    else:
                        log_test(f"{name} required columns", "PASS")
                    
                    # Check for nulls
                    null_counts = df.isnull().sum()
                    if null_counts.sum() > 0:
                        log_test(f"{name} data quality", "FAIL", 
                                f"Has {null_counts.sum()} null values", 
                                dict(null_counts[null_counts > 0]))
                    else:
                        log_test(f"{name} data quality", "PASS", "No nulls")
                    
                    # Check timestamp range
                    if 'timestamp' in df.columns:
                        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                        log_test(f"{name} temporal coverage", "PASS", 
                                f"{date_range} days of data")
                
            else:
                log_test(f"Data file: {name}", "FAIL", f"File not found: {filepath}")
    
    except Exception as e:
        log_test("Data generation tests", "FAIL", str(e))

# =============================================================================
# TEST 3: Feature Engineering
# =============================================================================
def test_feature_engineering():
    test_section("TEST 3: Feature Engineering")
    
    try:
        import pandas as pd
        
        if os.path.exists("data/ml_features.parquet"):
            df = pd.read_parquet("data/ml_features.parquet")
            
            log_test("Feature file exists", "PASS", f"{len(df)} samples")
            
            # Check for rolling features (per assignment)
            rolling_patterns = ["rolling", "trend", "mean", "std", "min", "max"]
            feature_cols = df.columns.tolist()
            
            rolling_features = [c for c in feature_cols 
                              if any(p in c.lower() for p in rolling_patterns)]
            
            if rolling_features:
                log_test("Rolling features", "PASS", 
                        f"Found {len(rolling_features)} rolling features",
                        rolling_features[:5])
            else:
                log_test("Rolling features", "FAIL", 
                        "No rolling window features found")
            
            # Check for time-based features
            time_patterns = ["since", "hours", "days", "duration"]
            time_features = [c for c in feature_cols 
                           if any(p in c.lower() for p in time_patterns)]
            
            if time_features:
                log_test("Time-based features", "PASS", 
                        f"Found {len(time_features)} time features",
                        time_features[:3])
            else:
                log_test("Time-based features", "FAIL", 
                        "No time-based features found")
            
            # Check target variable
            if 'soot_load_pct' in df.columns:
                log_test("Target variable", "PASS", "soot_load_pct exists")
                
                # Check target distribution
                target = df['soot_load_pct']
                log_test("Target range", 
                        "PASS" if target.min() >= 0 and target.max() <= 100 else "FAIL",
                        f"Range: {target.min():.2f}% - {target.max():.2f}%")
            else:
                log_test("Target variable", "FAIL", "soot_load_pct not found")
        
        else:
            log_test("Feature engineering", "FAIL", 
                    "ml_features.parquet not found")
    
    except Exception as e:
        log_test("Feature engineering tests", "FAIL", str(e))

# =============================================================================
# TEST 4: Model Artifacts
# =============================================================================
def test_model_artifacts():
    test_section("TEST 4: Model Artifacts")
    
    try:
        import joblib
        
        model_files = {
            "Model (baseline)": "models/soot_regressor_lgbm.joblib",
            "Model (optimized)": "models/soot_regressor_optuna.joblib",
            "Feature list": "models/feature_list.joblib",
            "Prediction intervals": "models/prediction_interval.joblib"
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                try:
                    artifact = joblib.load(filepath)
                    log_test(f"Model artifact: {name}", "PASS", 
                            f"Loaded successfully ({type(artifact).__name__})")
                except Exception as e:
                    log_test(f"Model artifact: {name}", "FAIL", 
                            f"Cannot load: {str(e)}")
            else:
                log_test(f"Model artifact: {name}", 
                        "SKIP" if "optuna" in filepath else "FAIL",
                        "File not found" if "optuna" not in filepath 
                        else "Optional file")
    
    except Exception as e:
        log_test("Model artifacts test", "FAIL", str(e))

# =============================================================================
# TEST 5: Model Performance
# =============================================================================
def test_model_performance():
    test_section("TEST 5: Model Performance")
    
    try:
        import pandas as pd
        import joblib
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        # Load test data
        if os.path.exists("data/ml_features.parquet"):
            df = pd.read_parquet("data/ml_features.parquet")
            
            # Load model
            model_path = "models/soot_regressor_lgbm.joblib"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                feature_list = joblib.load("models/feature_list.joblib")
                
                # Prepare test data (use last 20%)
                df_sorted = df.sort_values('timestamp')
                split_idx = int(len(df_sorted) * 0.8)
                test_df = df_sorted.iloc[split_idx:]
                
                X_test = test_df[feature_list]
                y_test = test_df['soot_load_pct']
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                log_test("Model MAE", 
                        "PASS" if mae < 5.0 else "FAIL",
                        f"{mae:.2f}% (expect < 5%)")
                
                log_test("Model RMSE", 
                        "PASS" if rmse < 7.0 else "FAIL",
                        f"{rmse:.2f}% (expect < 7%)")
                
                log_test("Model R²", 
                        "PASS" if r2 > 0.85 else "FAIL",
                        f"{r2:.3f} (expect > 0.85)")
                
                # Check for reasonable predictions
                if (y_pred.min() >= 0) and (y_pred.max() <= 100):
                    log_test("Prediction bounds", "PASS", 
                            f"Range: {y_pred.min():.1f}% - {y_pred.max():.1f}%")
                else:
                    log_test("Prediction bounds", "FAIL", 
                            f"Out of range: {y_pred.min():.1f}% - {y_pred.max():.1f}%")
            
            else:
                log_test("Model performance", "FAIL", "Model file not found")
        else:
            log_test("Model performance", "FAIL", "Feature data not found")
    
    except Exception as e:
        log_test("Model performance tests", "FAIL", str(e))

# =============================================================================
# TEST 6: API Structure
# =============================================================================
def test_api_structure():
    test_section("TEST 6: API Structure")
    
    try:
        # Check if API files exist
        api_files = [
            "src/api/main.py",
            "src/api/schemas.py",
            "run_api.py"
        ]
        
        for filepath in api_files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                
                log_test(f"API file: {filepath}", "PASS")
                
                # Check for required endpoints in main.py
                if "main.py" in filepath:
                    required_endpoints = [
                        "/predict/soot-load",
                        "/predict/batch",
                        "/model/info",
                        "/health"
                    ]
                    
                    for endpoint in required_endpoints:
                        if endpoint in content:
                            log_test(f"Endpoint defined: {endpoint}", "PASS")
                        else:
                            log_test(f"Endpoint defined: {endpoint}", "FAIL",
                                    "Endpoint not found in code")
            else:
                log_test(f"API file: {filepath}", "FAIL", "File not found")
    
    except Exception as e:
        log_test("API structure tests", "FAIL", str(e))

# =============================================================================
# TEST 7: Recommendation Logic
# =============================================================================
def test_recommendation_logic():
    test_section("TEST 7: Recommendation Logic")
    
    try:
        # Check if recommender exists
        recommender_path = "src/recommendation/recommender.py"
        
        if os.path.exists(recommender_path):
            log_test("Recommender file exists", "PASS")
            
            with open(recommender_path, 'r') as f:
                content = f.read()
            
            # Check for recommendation categories
            expected_categories = ["OK", "MONITOR", "PASSIVE_REGEN", 
                                 "ACTIVE_REGEN", "INSPECTION"]
            
            found_categories = []
            for cat in expected_categories:
                if cat in content:
                    found_categories.append(cat)
            
            if len(found_categories) >= 4:
                log_test("Recommendation categories", "PASS", 
                        f"Found: {', '.join(found_categories)}")
            else:
                log_test("Recommendation categories", "FAIL",
                        f"Only found: {', '.join(found_categories)}")
        
        else:
            log_test("Recommender file", "FAIL", "File not found")
    
    except Exception as e:
        log_test("Recommendation logic tests", "FAIL", str(e))

# =============================================================================
# TEST 8: Docker Configuration
# =============================================================================
def test_docker_config():
    test_section("TEST 8: Docker Configuration")
    
    try:
        if os.path.exists("Dockerfile"):
            with open("Dockerfile", 'r') as f:
                dockerfile_content = f.read()
            
            # Check for essential Dockerfile elements
            checks = {
                "Base image": "FROM python:",
                "Working directory": "WORKDIR",
                "Requirements copy": "requirements.txt",
                "Port exposure": "EXPOSE",
                "Entry command": "CMD"
            }
            
            for check_name, check_string in checks.items():
                if check_string in dockerfile_content:
                    log_test(f"Dockerfile: {check_name}", "PASS")
                else:
                    log_test(f"Dockerfile: {check_name}", "FAIL",
                            f"Missing: {check_string}")
        else:
            log_test("Dockerfile", "FAIL", "File not found")
    
    except Exception as e:
        log_test("Docker config tests", "FAIL", str(e))

# =============================================================================
# TEST 9: Edge Cases (if implemented)
# =============================================================================
def test_edge_cases():
    test_section("TEST 9: Edge Cases")
    
    try:
        edge_case_path = "tests/edge_cases/test_edge_cases.py"
        
        if os.path.exists(edge_case_path):
            log_test("Edge case tests exist", "PASS")
            
            with open(edge_case_path, 'r') as f:
                content = f.read()
            
            # Check for specific edge cases
            edge_cases = [
                "new_dpf",
                "post_regen",
                "missing",
                "sensor",
                "out_of_range",
                "stale"
            ]
            
            found = [case for case in edge_cases if case in content.lower()]
            
            log_test("Edge case coverage", 
                    "PASS" if len(found) >= 3 else "SKIP",
                    f"Found tests for: {', '.join(found)}" if found 
                    else "No specific edge cases found")
        else:
            log_test("Edge case tests", "SKIP", "Optional file not found")
    
    except Exception as e:
        log_test("Edge case tests", "FAIL", str(e))

# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================
def run_all_tests():
    print("\n" + "="*60)
    print(" DPF SOOT PREDICTION - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f" Working Directory: {os.getcwd()}")
    print("="*60)
    
    # Run all tests
    test_file_structure()
    test_data_generation()
    test_feature_engineering()
    test_model_artifacts()
    test_model_performance()
    test_api_structure()
    test_recommendation_logic()
    test_docker_config()
    test_edge_cases()
    
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
    print()
    
    # List all failures
    failures = [r for r in test_results if r['status'] == 'FAIL']
    if failures:
        print("\n❌ FAILED TESTS:")
        print("-" * 60)
        for f in failures:
            print(f"  • {f['test']}: {f['message']}")
    
    # Save detailed results
    with open('/home/claude/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: /home/claude/test_results.json")
    
    return passed, failed, skipped

if __name__ == "__main__":
    # Change to repo directory if provided
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
        if os.path.exists(repo_path):
            os.chdir(repo_path)
            print(f"Changed to directory: {repo_path}")
    
    passed, failed, skipped = run_all_tests()
    
    # Exit code based on results
    sys.exit(0 if failed == 0 else 1)

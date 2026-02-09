from fastapi import FastAPI
import pandas as pd
import joblib
from pathlib import Path
from typing import List

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
)

from src.recommendation.recommender import recommend_action
from src.monitoring.logger import log_prediction
from src.monitoring.drift import load_baseline_stats, detect_feature_drift
from src.models.uncertainty import load_interval, apply_interval
from src.feature_store.redis_store import ingest_telemetry, fetch_last_n
from src.realtime.feature_builder import build_features_from_window
from src.api.schemas import IngestRequest, PredictFromRawRequest

from fastapi import HTTPException

APP = FastAPI(title="DPF Soot Load Prediction API", version="1.0")

MODEL_PATH = Path("models/soot_regressor_optuna.joblib")
FEATURES_PATH = Path("models/soot_regressor_optuna_feature_list.joblib")

print("✅ MODEL_PATH:", MODEL_PATH.resolve())
print("✅ FEATURES_PATH:", FEATURES_PATH.resolve())

model = None
feature_list: List[str] = []
interval_params = None

@APP.on_event("startup")
def load_artifacts():
    global model, feature_list, baseline_stats, interval_params

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Feature list file not found: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)

    baseline_stats = load_baseline_stats()
    interval_params = load_interval()


print("✅ Loaded model type:", type(model))
print("✅ Loaded feature count:", len(feature_list))


@APP.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": len(feature_list),
    }


@APP.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_type="LightGBMRegressor",
        target="soot_load_pct",
        n_features=len(feature_list),
        features=feature_list,
    )


def _predict_from_features_dict(features: dict) -> float:
    """
    Convert dict -> dataframe aligned with training features.
    Since we don't have historical data in single-shot API,
    we fill engineered features with reasonable defaults.
    """
    # Start with provided features
    row = features.copy()
    
    # ============================================
    # FILL MISSING FEATURES WITH SMART DEFAULTS
    # ============================================
    
    # Get key values
    temp_pre = float(row.get("exhaust_temp_pre_dpf_c", 400))
    temp_post = float(row.get("exhaust_temp_post_dpf_c", temp_pre - 20))
    pressure = float(row.get("differential_pressure_kpa", 10))
    flow = float(row.get("exhaust_flow_rate", 150))
    speed = float(row.get("vehicle_speed_kmh", 50))
    load = float(row.get("engine_load_pct", 50))
    
    # ============================================
    # 🔥 CRITICAL FIX: Trip features based on pressure
    # ============================================
    # Higher pressure = vehicle has been operating longer
    # These are CUMULATIVE since last regen, not just current trip
    
    if pressure >= 25:
        # Very high pressure = long operation time
        row.setdefault("duration_min", 2400.0)  # 40 hours
        row.setdefault("distance_km", 800.0)     # ~800 km
        row.setdefault("stop_start_count", 150)
    elif pressure >= 20:
        row.setdefault("duration_min", 1800.0)  # 30 hours
        row.setdefault("distance_km", 600.0)
        row.setdefault("stop_start_count", 100)
    elif pressure >= 15:
        # 🔥 THIS IS YOUR CITY TRUCK CASE
        row.setdefault("duration_min", 1200.0)  # 20 hours of operation
        row.setdefault("distance_km", 400.0)     # 400 km traveled
        row.setdefault("stop_start_count", 80)   # Lots of stops (city)
    elif pressure >= 10:
        row.setdefault("duration_min", 600.0)   # 10 hours
        row.setdefault("distance_km", 300.0)
        row.setdefault("stop_start_count", 40)
    else:
        # Low pressure = recently regenerated
        row.setdefault("duration_min", 120.0)   # 2 hours
        row.setdefault("distance_km", 100.0)
        row.setdefault("stop_start_count", 10)
    
    # Maintenance features
    row.setdefault("regen_success", 0)
    row.setdefault("time_since_last_maint_min", row["duration_min"])  # Match duration
    
    # ============================================
    # 1) Basic derived features
    # ============================================
    row["temp_delta_dpf_c"] = temp_post - temp_pre
    row["pressure_norm"] = pressure / (flow + 1e-6)
    row["load_speed_ratio"] = load / (speed + 1e-3)
    
    # ============================================
    # 2) Rolling window features
    # ============================================
    is_city_driving = pressure >= 15 or (speed < 60 and load > 60)
    is_heavy_load = load > 75
    is_cold = temp_pre < 320

    for window in ["10min", "30min", "60min"]:
        row[f"temp_pre_roll_mean_{window}"] = temp_pre

        if window == "10min":
            if is_city_driving or is_heavy_load:
                row[f"temp_pre_roll_std_{window}"] = 12.0
            elif is_cold:
                row[f"temp_pre_roll_std_{window}"] = 8.0
            else:
                row[f"temp_pre_roll_std_{window}"] = 5.0
        elif window == "30min":
            if is_city_driving or is_heavy_load:
                row[f"temp_pre_roll_std_{window}"] = 15.0
            elif is_cold:
                row[f"temp_pre_roll_std_{window}"] = 12.0
            else:
                row[f"temp_pre_roll_std_{window}"] = 8.0
        else:  # 60min
            if is_city_driving or is_heavy_load:
                row[f"temp_pre_roll_std_{window}"] = 18.0
            elif is_cold:
                row[f"temp_pre_roll_std_{window}"] = 15.0
            else:
                row[f"temp_pre_roll_std_{window}"] = 12.0

        row[f"pressure_roll_mean_{window}"] = pressure

        if is_city_driving or is_cold:
            row[f"pressure_trend_{window}"] = 0.10
        else:
            row[f"pressure_trend_{window}"] = 0.0
    
    # ============================================
    # 3) High temperature streak
    # ============================================
    HIGH_TEMP = 380.0
    COLD_TEMP = 320.0

    if temp_pre >= HIGH_TEMP:
        row["high_temp_streak_min"] = min(20.0, (temp_pre - HIGH_TEMP) / 4)
    else:
        row["high_temp_streak_min"] = 0.0
    
    # ============================================
    # 4) Driving mode distribution
    # ============================================
    if speed < 5:
        row["pct_idle_60min"] = 0.8
        row["pct_city_60min"] = 0.15
        row["pct_highway_60min"] = 0.05
    elif speed < 60:
        row["pct_idle_60min"] = 0.15
        row["pct_city_60min"] = 0.75
        row["pct_highway_60min"] = 0.10
    else:  # highway
        row["pct_idle_60min"] = 0.02
        row["pct_city_60min"] = 0.10
        row["pct_highway_60min"] = 0.88
    
    # ============================================
    # 5) Regen-related features
    # ============================================
    row["is_regen_event"] = 1 if temp_pre > 500 else 0

    # Time since last regen (matches duration for consistency)
    if pressure >= 25:
        row["time_since_last_regen_min"] = 3600.0
    elif pressure >= 20:
        row["time_since_last_regen_min"] = 2800.0
    elif pressure >= 15:
        row["time_since_last_regen_min"] = 2000.0
    elif pressure >= 10:
        row["time_since_last_regen_min"] = 1200.0
    else:
        row["time_since_last_regen_min"] = 300.0
    
    # Apply multipliers
    multiplier = 1.0
    if is_cold:
        multiplier *= 1.5
    if is_city_driving and speed < 60:
        multiplier *= 1.3
    if is_heavy_load:
        multiplier *= 1.2
    
    row["time_since_last_regen_min"] = row["time_since_last_regen_min"] * multiplier
    
    # Regen opportunity score
    highway_factor = row["pct_highway_60min"]
    if row["high_temp_streak_min"] > 0:
        temp_factor = min(1.0, row["high_temp_streak_min"] / 20)
    else:
        temp_factor = 0.0
    
    row["regen_opportunity_score"] = 0.6 * highway_factor + 0.4 * temp_factor
    
    # ============================================
    # 6) One-hot encoded features
    # ============================================
    if speed < 5:
        pattern = "idle"
    elif speed < 60 and load > 70:
        pattern = "heavy_load"
    elif speed >= 60:
        pattern = "highway"
    else:
        pattern = "city"
    
    row["driving_pattern_city"] = 1 if pattern == "city" else 0
    row["driving_pattern_heavy_load"] = 1 if pattern == "heavy_load" else 0
    row["driving_pattern_highway"] = 1 if pattern == "highway" else 0
    row["driving_pattern_idle"] = 1 if pattern == "idle" else 0
    
    row["action_type_active"] = 0
    row["action_type_none"] = 1
    row["action_type_passive"] = 0
    
    # ============================================
    # Convert to DataFrame and predict
    # ============================================
    X = pd.DataFrame([row])
    X = pd.get_dummies(X, drop_first=False)
    X = X.reindex(columns=feature_list, fill_value=0)
    
    pred = float(model.predict(X)[0])
    
    return pred


@APP.post("/predict/soot-load", response_model=PredictResponse)
def predict_single(req: PredictRequest):
    validate_features(req.features)
    soot_pred = _predict_from_features_dict(req.features)

    low, high = apply_interval(soot_pred, interval_params)
    ci_level = 1.0 - float(interval_params["alpha"])

    row = pd.Series(req.features)
    row["soot_pred"] = soot_pred

    rec = recommend_action(row)

    drifted = detect_feature_drift(req.features, baseline_stats)

    log_prediction({
          "vehicle_id": req.vehicle_id,
          "timestamp": req.timestamp,
          "soot_pred_pct": soot_pred,
          "soot_pred_low_pct": low,
          "soot_pred_high_pct": high,
          "ci_level": ci_level,
          "recommended_action": rec["recommended_action"],
          "priority": rec["priority"],
          "reason": rec["reason"],
          "drifted_features": drifted,
          "n_drifted": len(drifted),
    })


    return PredictResponse(
          vehicle_id=req.vehicle_id,
          timestamp=req.timestamp,
          soot_pred_pct=soot_pred,
          soot_pred_low_pct=low,
          soot_pred_high_pct=high,
          ci_level=ci_level,
          recommended_action=rec["recommended_action"],
          priority=rec["priority"],
          reason=rec["reason"],
          drifted_features=drifted,
     )

@APP.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    results = []
    for item in req.items:
        validate_features(item.features)
        soot_pred = _predict_from_features_dict(item.features)

        # ✅ ADD: Calculate confidence intervals (was missing!)
        low, high = apply_interval(soot_pred, interval_params)
        ci_level = 1.0 - float(interval_params["alpha"])

        row = pd.Series(item.features)
        row["soot_pred"] = soot_pred
        
        rec = recommend_action(row)
        
        # ✅ ADD: Drift detection (optional but good)
        drifted = detect_feature_drift(item.features, baseline_stats)

        results.append(
            PredictResponse(
                vehicle_id=item.vehicle_id,
                timestamp=item.timestamp,
                soot_pred_pct=soot_pred,
                soot_pred_low_pct=low,           # ✅ FIXED: was missing
                soot_pred_high_pct=high,          # ✅ FIXED: was missing
                ci_level=ci_level,                # ✅ FIXED: was missing
                recommended_action=rec["recommended_action"],
                priority=rec["priority"],
                reason=rec["reason"],
                drifted_features=drifted,         # ✅ FIXED: was missing
            )
        )

    return BatchPredictResponse(results=results)





def validate_features(features: dict):
    required = [
        "engine_load_pct",
        "engine_rpm",
        "vehicle_speed_kmh",
        "exhaust_temp_pre_dpf_c",
        "differential_pressure_kpa",
    ]

    missing = [k for k in required if k not in features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    # Range checks (basic edge cases)
    if not (0 <= float(features["engine_load_pct"]) <= 100):
        raise HTTPException(status_code=422, detail="engine_load_pct out of range (0..100)")

    if not (0 <= float(features["vehicle_speed_kmh"]) <= 160):
        raise HTTPException(status_code=422, detail="vehicle_speed_kmh out of range (0..160)")

    if not (400 <= float(features["engine_rpm"]) <= 6000):
        raise HTTPException(status_code=422, detail="engine_rpm out of range (400..6000)")

    if not (-40 <= float(features.get("ambient_temp_c", 25)) <= 60):
        raise HTTPException(status_code=422, detail="ambient_temp_c out of range (-40..60)")

    if not (0 <= float(features["differential_pressure_kpa"]) <= 200):
        raise HTTPException(status_code=422, detail="differential_pressure_kpa out of range (0..200)")


@APP.post("/ingest/telemetry")
def ingest(req: IngestRequest):
    ingest_telemetry(req.vehicle_id, req.record.model_dump(), keep_last=120)
    return {"status": "ok", "vehicle_id": req.vehicle_id}


@APP.post("/predict/from-raw", response_model=PredictResponse)
def predict_from_raw(req: PredictFromRawRequest):
    window = fetch_last_n(req.vehicle_id, n=req.last_n)

    if len(window) < 10:
        raise HTTPException(status_code=422, detail="Not enough telemetry history to predict")

    engineered = build_features_from_window(window)

    # ✅ reuse existing pipeline
    validate_features(engineered)
    soot_pred = _predict_from_features_dict(engineered)

    low, high = apply_interval(soot_pred, interval_params)
    ci_level = 1.0 - float(interval_params["alpha"])

    row = pd.Series(engineered)
    row["soot_pred"] = soot_pred
    rec = recommend_action(row)

    drifted = detect_feature_drift(engineered, baseline_stats)

    log_prediction({
        "vehicle_id": req.vehicle_id,
        "timestamp": window[-1]["timestamp"],
        "soot_pred_pct": soot_pred,
        "soot_pred_low_pct": low,
        "soot_pred_high_pct": high,
        "ci_level": ci_level,
        "recommended_action": rec["recommended_action"],
        "priority": rec["priority"],
        "reason": rec["reason"],
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "mode": "from_raw_redis"
    })

    return PredictResponse(
        vehicle_id=req.vehicle_id,
        timestamp=window[-1]["timestamp"],
        soot_pred_pct=soot_pred,
        soot_pred_low_pct=low,
        soot_pred_high_pct=high,
        ci_level=ci_level,
        recommended_action=rec["recommended_action"],
        priority=rec["priority"],
        reason=rec["reason"],
        drifted_features=drifted,
    )

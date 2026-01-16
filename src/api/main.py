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
    Missing features will be filled with 0.
    Extra features will be ignored.
    """
    X = pd.DataFrame([features])

    # convert object columns to one-hot (safe)
    X = pd.get_dummies(X, drop_first=False)

    # align with training features
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

        row = pd.Series(item.features)
        row["soot_pred"] = soot_pred
        rec = recommend_action(row)

        results.append(
            PredictResponse(
                vehicle_id=item.vehicle_id,
                timestamp=item.timestamp,
                soot_pred_pct=soot_pred,
                recommended_action=rec["recommended_action"],
                priority=rec["priority"],
                reason=rec["reason"],
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

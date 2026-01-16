import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error

from src.models.train_regression import TrainConfig, prepare_features

API_URL = "http://localhost:8000/predict/soot-load"

def main():
    cfg = TrainConfig()

    # ✅ Load full dataset
    df_full = pd.read_parquet("data/processed/ml_features.parquet")

    # ✅ Load optuna feature list
    feature_list = joblib.load("models/soot_regressor_optuna_feature_list.joblib")

    # ✅ Prepare features on FULL df (this is the critical fix)
    X_full, y_full = prepare_features(df_full, cfg)
    X_full = X_full.reindex(columns=feature_list, fill_value=0)

    # ✅ Now sample rows AFTER preprocessing
    sample_idx = df_full.sample(50, random_state=42).index
    sample_raw = df_full.loc[sample_idx].reset_index(drop=True)
    X_sample = X_full.loc[sample_idx].reset_index(drop=True)

    y_true = []
    y_pred = []

    for i in range(len(sample_raw)):
        features = X_sample.iloc[i].to_dict()

        payload = {
            "vehicle_id": str(sample_raw.loc[i, "vehicle_id"]),
            "timestamp": str(sample_raw.loc[i, "timestamp"]),
            "features": features
        }

        resp = requests.post(API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        out = resp.json()

        y_true.append(float(sample_raw.loc[i, "soot_load_pct"]))
        y_pred.append(float(out["soot_pred_pct"]))

    mae = mean_absolute_error(y_true, y_pred)

    print("✅ FINAL LIVE API EVALUATION (FULL PREPROCESS + SAMPLE)")
    print("Samples:", len(y_true))
    print("MAE:", round(mae, 5))

    if mae < 0.22:
        print("✅ PASS: API matches tuned model performance.")
    elif mae < 0.26:
        print("⚠️ OK: Slightly higher but acceptable.")
    else:
        print("❌ FAIL: Still misaligned — unlikely now.")

if __name__ == "__main__":
    main()

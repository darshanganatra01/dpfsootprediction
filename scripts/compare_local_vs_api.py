import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import requests

from src.models.train_regression import TrainConfig, prepare_features

API_URL = "http://localhost:8000/predict/soot-load"

def main():
    cfg = TrainConfig()
    df = pd.read_parquet("data/processed/ml_features.parquet").sample(1, random_state=7).reset_index(drop=True)

    model = joblib.load("models/soot_regressor_optuna.joblib")
    feature_list = joblib.load("models/soot_regressor_optuna_feature_list.joblib")

    X, y = prepare_features(df, cfg)
    X = X.reindex(columns=feature_list, fill_value=0)

    features = X.iloc[0].to_dict()
    payload = {
        "vehicle_id": str(df.loc[0, "vehicle_id"]),
        "timestamp": str(df.loc[0, "timestamp"]),
        "features": features
    }

    # Local pred
    local_pred = float(model.predict(pd.DataFrame([features]).reindex(columns=feature_list, fill_value=0))[0])

    # API pred
    api_resp = requests.post(API_URL, json=payload, timeout=30)
    api_resp.raise_for_status()
    api_pred = float(api_resp.json()["soot_pred_pct"])

    print("✅ Local prediction:", local_pred)
    print("✅ API prediction  :", api_pred)
    print("✅ Abs diff        :", abs(local_pred - api_pred))

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from pathlib import Path

BASELINE_FILE = Path("models/baseline_feature_stats.parquet")

def compute_baseline_stats(train_df: pd.DataFrame, feature_cols: list):
    """
    Creates baseline stats from TRAIN data only.
    Handles bool/int/float columns safely.
    """
    stats = []

    for col in feature_cols:
        if col not in train_df.columns:
            continue

        s = train_df[col]

        # ✅ Fix: convert boolean to int (otherwise quantile breaks)
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(int)

        # only numeric columns
        if not pd.api.types.is_numeric_dtype(s):
            continue

        # convert to float for safety
        s = s.astype(float)

        std = float(s.std())
        if std <= 1e-9 or np.isnan(std):
            std = 1e-9

        stats.append({
            "feature": col,
            "mean": float(s.mean()),
            "std": std,
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
        })

    out = pd.DataFrame(stats)
    out.to_parquet(BASELINE_FILE, index=False)
    return out


def load_baseline_stats() -> pd.DataFrame:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline stats not found at {BASELINE_FILE}. Generate it during training."
        )
    return pd.read_parquet(BASELINE_FILE)



def drift_score_z(current_value: float, mean: float, std: float) -> float:
    return abs(current_value - mean) / std


def detect_feature_drift(features: dict, baseline_stats: pd.DataFrame, z_thresh: float = 3.0):
    """
    Very simple drift detection:
    if any feature deviates > z_thresh std from baseline mean -> flag drift
    """
    drifted = []

    baseline_map = baseline_stats.set_index("feature").to_dict(orient="index")
    for feat, val in features.items():
        if feat not in baseline_map:
            continue
        try:
            val = float(val)
        except:
            continue

        mean = baseline_map[feat]["mean"]
        std = baseline_map[feat]["std"]
        z = drift_score_z(val, mean, std)

        if z >= z_thresh:
            drifted.append({
                "feature": feat,
                "value": val,
                "z_score": float(z)
            })

    return drifted

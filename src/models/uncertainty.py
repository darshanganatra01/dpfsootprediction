import numpy as np
import joblib
from pathlib import Path

CI_PATH = Path("models/prediction_interval.joblib")

def fit_residual_interval(y_true, y_pred, alpha=0.10):
    """
    Residual-based prediction interval.
    alpha=0.10 => 90% PI
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    low_q = alpha / 2
    high_q = 1 - alpha / 2

    q_low = float(np.quantile(residuals, low_q))
    q_high = float(np.quantile(residuals, high_q))

    return {
        "method": "residual_quantile",
        "alpha": float(alpha),
        "q_low": q_low,
        "q_high": q_high,
        "low_q": low_q,
        "high_q": high_q,
    }

def apply_interval(pred, interval_params):
    q_low = interval_params["q_low"]
    q_high = interval_params["q_high"]
    return float(pred + q_low), float(pred + q_high)

def save_interval(interval_params):
    CI_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(interval_params, CI_PATH)

def load_interval():
    if not CI_PATH.exists():
        raise FileNotFoundError(f"Prediction interval file not found: {CI_PATH}")
    return joblib.load(CI_PATH)

def coverage_score(y_true, y_pred, interval_params):
    """
    What percent of true values are inside the interval?
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    low = y_pred + interval_params["q_low"]
    high = y_pred + interval_params["q_high"]

    inside = (y_true >= low) & (y_true <= high)
    return float(inside.mean())

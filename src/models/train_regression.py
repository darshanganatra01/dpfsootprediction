from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.uncertainty import fit_residual_interval, save_interval, coverage_score


@dataclass
class TrainConfig:
    target_col: str = "soot_load_pct"
    timestamp_col: str = "timestamp"
    vehicle_col: str = "vehicle_id"

    test_size: float = 0.15
    val_size: float = 0.15

    model_out_path: str = "models/soot_regressor_lgbm.joblib"
    feature_list_out_path: str = "models/feature_list.joblib"


def time_split(df: pd.DataFrame, cfg: TrainConfig):
    """
    Split by time globally (not random) to prevent leakage.
    """
    df = df.sort_values(cfg.timestamp_col).reset_index(drop=True)
    n = len(df)

    n_test = int(n * cfg.test_size)
    n_val = int(n * cfg.val_size)

    train = df.iloc[: n - n_val - n_test]
    val = df.iloc[n - n_val - n_test : n - n_test]
    test = df.iloc[n - n_test :]

    return train, val, test


def prepare_features(df: pd.DataFrame, cfg: TrainConfig):
    """
    Drop non-feature columns and handle NaNs safely.
    """
    df = df.copy()

    # Ensure timestamp type
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])

    # Drop columns that should NOT be used as raw features
    drop_cols = [
        cfg.target_col,
        "notes",           # text
        "maint_ts",        # timestamp of maintenance event (can leak/hard to model)
    ]

    # Optional: drop raw action_type (string) OR encode it.
    # We’ll use safe encoding:
    # missing => "none"
    if "action_type" in df.columns:
        df["action_type"] = df["action_type"].fillna("none")

    # regen_success sometimes missing
    if "regen_success" in df.columns:
        df["regen_success"] = df["regen_success"].fillna(False).astype(int)

    # time_since_last_regen_min is a core feature but has NaNs early -> fill with large number
    if "time_since_last_regen_min" in df.columns:
        df["time_since_last_regen_min"] = df["time_since_last_regen_min"].fillna(1e6)

    # Same for maint mins
    if "time_since_last_maint_min" in df.columns:
        df["time_since_last_maint_min"] = df["time_since_last_maint_min"].fillna(1e6)

    # Remaining rolling NaNs -> fill using forward fill per vehicle, then global fill
    # This preserves time ordering (no future leakage)
    df = df.sort_values([cfg.vehicle_col, cfg.timestamp_col])
    df = df.groupby(cfg.vehicle_col).apply(lambda g: g.ffill()).reset_index(drop=True)
    df = df.fillna(0)

    # Now separate X/y
    y = df[cfg.target_col].astype(float)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Remove identifiers
    for col in [cfg.vehicle_col, cfg.timestamp_col, "trip_id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype == "object"], drop_first=False)

    return X, y


def train_lgbm_regressor(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )

    return model


def evaluate_split(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"\n📌 {name} Metrics")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    return mae, rmse


def train_pipeline(df: pd.DataFrame, cfg: TrainConfig):
    train_df, val_df, test_df = time_split(df, cfg)

    print("✅ Split sizes:")
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    X_train, y_train = prepare_features(train_df, cfg)
    X_val, y_val = prepare_features(val_df, cfg)
    X_test, y_test = prepare_features(test_df, cfg)

    # Align columns (very important after one-hot)
    X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


    print("\n✅ Final feature count:", X_train.shape[1])

    # ✅ NEW: save baseline stats for drift detection
    from src.monitoring.drift import compute_baseline_stats
    compute_baseline_stats(train_df=X_train, feature_cols=list(X_train.columns))
    print("✅ Saved baseline feature stats for drift detection.")	

    model = train_lgbm_regressor(X_train, y_train, X_val, y_val)

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    evaluate_split("Validation", y_val, pred_val)
    evaluate_split("Test", y_test, pred_test)

    # ✅ CI fitting (from validation residuals)
    interval_params = fit_residual_interval(y_true=y_val, y_pred=pred_val, alpha=0.10)  # 90% PI
    save_interval(interval_params)

    cov = coverage_score(y_true=y_val, y_pred=pred_val, interval_params=interval_params)
    print("\n✅ Confidence Interval saved: models/prediction_interval.joblib")
    print(f"✅ 90% PI coverage on validation: {cov*100:.2f}%")


    # Save artifacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, cfg.model_out_path)
    joblib.dump(list(X_train.columns), cfg.feature_list_out_path)

    print("\n✅ Saved:")
    print(" -", cfg.model_out_path)
    print(" -", cfg.feature_list_out_path)

    return model

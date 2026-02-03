import mlflow
import mlflow.sklearn

mlflow.set_experiment("DPF_Soot_Load_Prediction")

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.uncertainty import (
    fit_residual_interval,
    save_interval,
    coverage_score
)
from src.monitoring.drift import compute_baseline_stats


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
    df[cfg.timestamp_col] = pd.to_datetime(df[cfg.timestamp_col])

    drop_cols = [
        cfg.target_col,
        "notes",
        "maint_ts",
    ]

    if "action_type" in df.columns:
        df["action_type"] = df["action_type"].fillna("none")

    if "regen_success" in df.columns:
        df["regen_success"] = df["regen_success"].fillna(False).astype(int)

    if "time_since_last_regen_min" in df.columns:
        df["time_since_last_regen_min"] = df["time_since_last_regen_min"].fillna(1e6)

    if "time_since_last_maint_min" in df.columns:
        df["time_since_last_maint_min"] = df["time_since_last_maint_min"].fillna(1e6)

    df = df.sort_values([cfg.vehicle_col, cfg.timestamp_col])
    df = df.groupby(cfg.vehicle_col).apply(lambda g: g.ffill()).reset_index(drop=True)
    df = df.fillna(0)

    y = df[cfg.target_col].astype(float)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in [cfg.vehicle_col, cfg.timestamp_col, "trip_id"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X = pd.get_dummies(
        X,
        columns=[c for c in X.columns if X[c].dtype == "object"],
        drop_first=False
    )

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

    with mlflow.start_run(run_name="baseline_lgbm"):

        train_df, val_df, test_df = time_split(df, cfg)

        X_train, y_train = prepare_features(train_df, cfg)
        X_val, y_val = prepare_features(val_df, cfg)
        X_test, y_test = prepare_features(test_df, cfg)

        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        compute_baseline_stats(train_df=X_train, feature_cols=list(X_train.columns))

        model = train_lgbm_regressor(X_train, y_train, X_val, y_val)

        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

        val_mae, val_rmse = evaluate_split("Validation", y_val, pred_val)
        test_mae, test_rmse = evaluate_split("Test", y_test, pred_test)

        interval_params = fit_residual_interval(
            y_true=y_val,
            y_pred=pred_val,
            alpha=0.10
        )
        save_interval(interval_params)

        coverage = coverage_score(
            y_true=y_val,
            y_pred=pred_val,
            interval_params=interval_params
        )

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, cfg.model_out_path)
        joblib.dump(list(X_train.columns), cfg.feature_list_out_path)

        mlflow.log_params({
            "model_type": "LightGBM",
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("pi_coverage_90", coverage)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(cfg.feature_list_out_path)
        mlflow.log_artifact("models/prediction_interval.joblib")

        return model

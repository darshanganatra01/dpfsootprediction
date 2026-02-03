import mlflow
import mlflow.sklearn

mlflow.set_experiment("DPF_Soot_Load_Prediction")

import optuna
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from src.models.train_regression import (
    TrainConfig,
    time_split,
    prepare_features
)


def objective(trial: optuna.Trial, df: pd.DataFrame, cfg: TrainConfig) -> float:

    with mlflow.start_run(nested=True):

        train_df, val_df, _ = time_split(df, cfg)

        X_train, y_train = prepare_features(train_df, cfg)
        X_val, y_val = prepare_features(val_df, cfg)

        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)

        mlflow.log_params(params)
        mlflow.log_metric("val_mae", mae)
        mlflow.set_tag("optuna_trial", trial.number)

        return mae


def run_optuna_hpt(df: pd.DataFrame, cfg: TrainConfig, n_trials: int = 30):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, df, cfg), n_trials=n_trials)
    return study


def train_best_model(df: pd.DataFrame, cfg: TrainConfig, best_params: dict):

    with mlflow.start_run(run_name="optuna_best_model"):

        train_df, val_df, test_df = time_split(df, cfg)

        X_train, y_train = prepare_features(train_df, cfg)
        X_val, y_val = prepare_features(val_df, cfg)
        X_test, y_test = prepare_features(test_df, cfg)

        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )

        pred_test = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, pred_test)

        mlflow.log_params(best_params)
        mlflow.log_metric("test_mae", mae_test)
        mlflow.sklearn.log_model(model, "model")

        return model, list(X_train.columns), mae_test


def save_hpt_outputs(
    study,
    model,
    feature_list,
    out_prefix="models/soot_regressor_optuna"
):
    Path("models").mkdir(exist_ok=True)

    joblib.dump(study.best_params, f"{out_prefix}_best_params.joblib")
    joblib.dump(model, f"{out_prefix}.joblib")
    joblib.dump(feature_list, f"{out_prefix}_feature_list.joblib")

    print("\n✅ Saved tuned artifacts:")
    print(f" - {out_prefix}.joblib")
    print(f" - {out_prefix}_feature_list.joblib")
    print(f" - {out_prefix}_best_params.joblib")

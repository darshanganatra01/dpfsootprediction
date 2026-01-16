import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.models.train_regression import TrainConfig, time_split, prepare_features
from src.models.uncertainty import fit_residual_interval, save_interval, coverage_score

def main():
    cfg = TrainConfig()
    df = pd.read_parquet("data/processed/ml_features.parquet")

    # time split
    train_df, val_df, test_df = time_split(df, cfg)

    # load optuna model + features
    model = joblib.load("models/soot_regressor_optuna.joblib")
    feature_list = joblib.load("models/soot_regressor_optuna_feature_list.joblib")

    # prepare val
    X_val, y_val = prepare_features(val_df, cfg)
    X_val = X_val.reindex(columns=feature_list, fill_value=0)

    pred_val = model.predict(X_val)
    mae_val = mean_absolute_error(y_val, pred_val)

    # fit CI based on tuned model residuals
    interval_params = fit_residual_interval(y_true=y_val, y_pred=pred_val, alpha=0.10)
    save_interval(interval_params)

    cov = coverage_score(y_true=y_val, y_pred=pred_val, interval_params=interval_params)

    print("✅ Optuna model validation MAE:", round(mae_val, 5))
    print("✅ Saved CI file: models/prediction_interval.joblib")
    print(f"✅ 90% PI coverage on validation: {cov*100:.2f}%")

if __name__ == "__main__":
    main()

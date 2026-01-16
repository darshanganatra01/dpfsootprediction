import pandas as pd
import joblib

from src.models.train_regression import TrainConfig, prepare_features
from src.recommendation.recommender import apply_recommendations

def main():
    # Load data
    df = pd.read_parquet("data/processed/ml_features.parquet")

    # Load model + features
    model = joblib.load("models/soot_regressor_lgbm.joblib")
    feature_list = joblib.load("models/feature_list.joblib")

    # Prepare X from full df
    cfg = TrainConfig()
    X, y = prepare_features(df, cfg)

    # align columns
    X = X.reindex(columns=feature_list, fill_value=0)

    # predict
    df_out = df[["vehicle_id", "timestamp", "soot_load_pct", "regen_opportunity_score", "differential_pressure_kpa"]].copy()
    df_out["soot_pred"] = model.predict(X)

    # recommendations
    df_out = apply_recommendations(df_out)

    print(df_out.head(25))
    print("\nAction counts:")
    print(df_out["recommended_action"].value_counts())

if __name__ == "__main__":
    main()


import pandas as pd
from src.models.train_regression import TrainConfig
from src.models.hpt_optuna import run_optuna_hpt, train_best_model, save_hpt_outputs

def main():
    df = pd.read_parquet("data/processed/ml_features.parquet")

    cfg = TrainConfig()

    # 🔧 tune trials (start small)
    n_trials = 20

    print(f"🚀 Starting Optuna HPT with {n_trials} trials...")
    study = run_optuna_hpt(df, cfg, n_trials=n_trials)

    print("\n✅ Best Optuna Params:")
    print(study.best_params)
    print("✅ Best Validation MAE:", study.best_value)

    # Train final tuned model using best params
    model, feature_list, mae_test = train_best_model(df, cfg, study.best_params)

    print("\n📌 Tuned model Test MAE:", mae_test)

    save_hpt_outputs(study, model, feature_list)

if __name__ == "__main__":
    main()

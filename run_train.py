import pandas as pd
from src.models.train_regression import TrainConfig, train_pipeline

def main():
    df = pd.read_parquet("data/processed/ml_features.parquet")
    cfg = TrainConfig()
    train_pipeline(df, cfg)

if __name__ == "__main__":
    main()

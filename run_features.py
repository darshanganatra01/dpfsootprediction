from pathlib import Path
import pandas as pd

from src.features.build_features import build_features

PROC_DIR = Path("data/processed")

def main():
    df = pd.read_parquet(PROC_DIR / "ml_base_table.parquet")
    out = build_features(df)

    out.to_parquet(PROC_DIR / "ml_features.parquet", index=False)

    print("✅ Feature dataset saved:")
    print(" - data/processed/ml_features.parquet")
    print("Rows:", len(out), "Cols:", out.shape[1])

if __name__ == "__main__":
    main()

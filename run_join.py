from pathlib import Path
import pandas as pd
from src.data_engineering.join import join_tables

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    sensor = pd.read_parquet(RAW_DIR / "sensor_telemetry.parquet")
    trips = pd.read_parquet(RAW_DIR / "trip_characteristics.parquet")
    maint = pd.read_parquet(RAW_DIR / "maintenance_records.parquet")

    df = join_tables(sensor, trips, maint)
    df.to_parquet(PROC_DIR / "ml_base_table.parquet", index=False)

    print("✅ Saved processed table: data/processed/ml_base_table.parquet")
    print("Rows:", len(df), "Cols:", df.shape[1])

if __name__ == "__main__":
    main()

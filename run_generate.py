from pathlib import Path
from src.data_generation.config import SimConfig
from src.data_generation.generate_all import generate_all

RAW_DIR = Path("data/raw")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(
        n_vehicles=40,
        n_days=45,
        start_date="2025-11-01",
        telemetry_freq="1min",
    )

    vehicles, trips, telemetry, maintenance = generate_all(cfg)

    vehicles.to_parquet(RAW_DIR / "vehicles.parquet", index=False)
    trips.to_parquet(RAW_DIR / "trip_characteristics.parquet", index=False)
    telemetry.to_parquet(RAW_DIR / "sensor_telemetry.parquet", index=False)
    maintenance.to_parquet(RAW_DIR / "maintenance_records.parquet", index=False)

    print("✅ Generated datasets:")
    print(" - data/raw/vehicles.parquet")
    print(" - data/raw/trip_characteristics.parquet")
    print(" - data/raw/sensor_telemetry.parquet")
    print(" - data/raw/maintenance_records.parquet")

if __name__ == "__main__":
    main()

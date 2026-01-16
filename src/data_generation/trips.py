import numpy as np
import pandas as pd
from .config import SimConfig

def _choose_trip_pattern(rng, vehicle_class: str) -> str:
    if vehicle_class == "highway":
        return rng.choice(["highway", "city"], p=[0.8, 0.2])
    if vehicle_class == "city":
        return rng.choice(["city", "idle", "highway"], p=[0.65, 0.2, 0.15])
    return rng.choice(["heavy_load", "city"], p=[0.7, 0.3])

def generate_trips(cfg: SimConfig, vehicles: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)

    start_ts = pd.Timestamp(cfg.start_date)
    trips = []

    for _, v in vehicles.iterrows():
        for d in range(cfg.n_days):
            day_start = start_ts + pd.Timedelta(days=d)

            n_trips = int(rng.integers(cfg.min_trips_per_day, cfg.max_trips_per_day + 1))
            trip_start = day_start + pd.Timedelta(minutes=int(rng.integers(0, 180)))

            for t in range(n_trips):
                pattern = _choose_trip_pattern(rng, v["vehicle_class"])

                # skewed durations
                if pattern == "highway":
                    duration_min = int(rng.gamma(8, 12))
                elif pattern == "city":
                    duration_min = int(rng.gamma(6, 10))
                elif pattern == "idle":
                    duration_min = int(rng.gamma(3, 10))
                else:
                    duration_min = int(rng.gamma(7, 11))

                duration_min = int(np.clip(duration_min, 15, 240))
                end_time = trip_start + pd.Timedelta(minutes=duration_min)

                # stop-start behavior
                if pattern == "highway":
                    stop_starts = int(rng.poisson(2))
                elif pattern == "city":
                    stop_starts = int(rng.poisson(18))
                elif pattern == "idle":
                    stop_starts = int(rng.poisson(6))
                else:
                    stop_starts = int(rng.poisson(10))

                avg_speed = {
                    "highway": rng.normal(75, 8),
                    "city": rng.normal(25, 10),
                    "idle": rng.normal(3, 2),
                    "heavy_load": rng.normal(30, 8),
                }[pattern]
                avg_speed = max(0, avg_speed)
                distance_km = max(0, avg_speed * (duration_min / 60))

                trips.append({
                    "trip_id": f"{v.vehicle_id}_D{d:03d}_T{t}",
                    "vehicle_id": v.vehicle_id,
                    "start_time": trip_start,
                    "end_time": end_time,
                    "duration_min": duration_min,
                    "distance_km": distance_km,
                    "stop_start_count": stop_starts,
                    "driving_pattern": pattern,
                })

                # gap
                trip_start = end_time + pd.Timedelta(minutes=int(rng.integers(30, 240)))

    return pd.DataFrame(trips).sort_values(["vehicle_id", "start_time"]).reset_index(drop=True)

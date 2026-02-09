from dataclasses import dataclass

@dataclass
class SimConfig:
    seed: int = 42

    n_vehicles: int = 50
    start_date: str = "2025-11-01"
    n_days: int = 60

    telemetry_freq: str = "1min"      # 1 row per minute
    min_trips_per_day: int = 2
    max_trips_per_day: int = 5

    # thresholds used in simulation
    passive_temp_thresh: float = 320.0   # C (increased back - only highway at very high temp)
    passive_speed_thresh: float = 80.0   # km/h (increased - harder to get passive regen)
    passive_window_min: int = 10

    active_soot_thresh: float = 75.0     # % (was 80 - trigger earlier)
    active_pressure_thresh: float = 25.0 # kPa

    # noise/missingness
    random_missing_pct: float = 0.02
    outlier_pct: float = 0.02

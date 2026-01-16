import numpy as np
import pandas as pd
from .config import SimConfig

def _generate_speed_series(rng, pattern: str, n: int) -> np.ndarray:
    speed = np.zeros(n)
    for i in range(n):
        prev = speed[i-1] if i > 0 else 0.0

        if pattern == "highway":
            if i == 0:
                speed[i] = rng.normal(70, 5)
            else:
                speed[i] = np.clip(prev + rng.normal(0, 2), 50, 105)

        elif pattern == "city":
            if rng.random() < 0.25:
                speed[i] = 0
            else:
                speed[i] = np.clip(rng.normal(25, 10), 0, 65)

        elif pattern == "idle":
            speed[i] = np.clip(rng.normal(1, 1), 0, 5)

        else:  # heavy_load
            if i == 0:
                speed[i] = rng.normal(20, 6)
            else:
                speed[i] = np.clip(prev + rng.normal(0, 3), 5, 55)

    return speed

def generate_telemetry_for_trip(cfg: SimConfig, trip_row: pd.Series, vehicle_row: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 10)

    ts = pd.date_range(trip_row.start_time, trip_row.end_time, freq=cfg.telemetry_freq, inclusive="left")
    n = len(ts)

    speed = _generate_speed_series(rng, trip_row.driving_pattern, n)
    accel = np.diff(speed, prepend=speed[0])

    hour = np.array([t.hour + t.minute / 60 for t in ts])
    ambient = 25 + 5*np.sin(2*np.pi * hour/24) + rng.normal(0, 1, size=n)

    base_load = {"highway": 30, "city": 20, "idle": 10, "heavy_load": 55}[trip_row.driving_pattern]
    load = base_load + 0.45*speed + 1.2*np.maximum(accel, 0) + rng.normal(0, 6, size=n)
    load = np.clip(load, 5, 100)

    rpm = 800 + 20*speed + 8*load + rng.normal(0, 60, size=n)
    rpm = np.clip(rpm, 600, 2500)

    flow = 40 + 0.04*rpm + 0.7*load + rng.normal(0, 5, size=n)
    flow = np.clip(flow, 10, None)

    temp_pre = 150 + 1.2*load + 0.05*rpm + 0.2*ambient + rng.normal(0, 10, size=n)
    temp_pre += vehicle_row.temp_sensor_bias
    temp_pre = np.clip(temp_pre, 80, 700)

    temp_post = temp_pre - rng.normal(5, 3, size=n)

    df = pd.DataFrame({
        "vehicle_id": trip_row.vehicle_id,
        "trip_id": trip_row.trip_id,
        "timestamp": ts,
        "vehicle_speed_kmh": speed,
        "engine_load_pct": load,
        "engine_rpm": rpm,
        "ambient_temp_c": ambient,
        "exhaust_flow_rate": flow,
        "exhaust_temp_pre_dpf_c": temp_pre,
        "exhaust_temp_post_dpf_c": temp_post,
        "driving_pattern": trip_row.driving_pattern,
    })
    return df

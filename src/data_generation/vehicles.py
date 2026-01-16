import numpy as np
import pandas as pd
from .config import SimConfig

def generate_vehicles(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    vehicle_ids = [f"V{str(i).zfill(3)}" for i in range(1, cfg.n_vehicles + 1)]
    vehicle_class = rng.choice(
        ["city", "highway", "heavy_load"], size=cfg.n_vehicles, p=[0.4, 0.4, 0.2]
    )

    df = pd.DataFrame({
        "vehicle_id": vehicle_ids,
        "vehicle_class": vehicle_class,
        "regen_efficiency": rng.beta(8, 2, size=cfg.n_vehicles),   # mostly good vehicles
        "temp_sensor_bias": rng.normal(0, 2.0, size=cfg.n_vehicles),
        "press_sensor_bias": rng.normal(0, 0.5, size=cfg.n_vehicles),
    })
    return df

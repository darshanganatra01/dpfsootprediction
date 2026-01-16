import numpy as np
import pandas as pd
from .config import SimConfig

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_soot_and_maintenance(cfg: SimConfig, telemetry_df: pd.DataFrame, vehicle_row: pd.Series):
    rng = np.random.default_rng(cfg.seed + 20)

    soot_state = float(rng.uniform(5, 25))
    cooldown = 0

    soot_vals = []
    maint_rows = []

    temp_window = []
    speed_window = []

    for _, row in telemetry_df.iterrows():
        temp_window.append(row.exhaust_temp_pre_dpf_c)
        speed_window.append(row.vehicle_speed_kmh)
        if len(temp_window) > cfg.passive_window_min:
            temp_window.pop(0)
            speed_window.pop(0)

        rolling_temp = float(np.mean(temp_window))
        rolling_speed = float(np.mean(speed_window))

        # accumulation
        city_bonus = 0.02 if row.driving_pattern == "city" else 0.0
        idle_bonus = 0.03 if row.vehicle_speed_kmh < 5 else 0.0
        accum = 0.01 * row.engine_load_pct + city_bonus + idle_bonus

        # burnoff
        burnoff = 0.05 * _sigmoid((row.exhaust_temp_pre_dpf_c - 320) / 20)

        soot_state = soot_state + accum - burnoff

        # passive regen condition
        passive_ok = (
            (rolling_temp > cfg.passive_temp_thresh) and
            (rolling_speed > cfg.passive_speed_thresh) and
            (row.driving_pattern == "highway")
        )

        did_passive = passive_ok and (rng.random() < 0.25)

        # active regen condition
        active_needed = (soot_state > cfg.active_soot_thresh) and (cooldown <= 0)
        did_active = active_needed and (rng.random() < 0.80)

        # apply regen effects
        if did_passive:
            drop = rng.normal(10, 3)
            soot_state -= drop
            maint_rows.append({
                "vehicle_id": row.vehicle_id,
                "timestamp": row.timestamp,
                "action_type": "passive",
                "regen_success": True,
                "notes": "passive regen during highway high-temp"
            })

        if did_active:
            success = rng.random() > 0.08
            drop = rng.normal(40, 8) * float(vehicle_row.regen_efficiency)
            soot_state -= (drop if success else drop * 0.3)
            cooldown = 120

            maint_rows.append({
                "vehicle_id": row.vehicle_id,
                "timestamp": row.timestamp,
                "action_type": "active",
                "regen_success": success,
                "notes": "active regen triggered by high soot"
            })

        cooldown -= 1
        soot_state = float(np.clip(soot_state, 0, 100))
        soot_vals.append(soot_state)

    telemetry_df = telemetry_df.copy()
    telemetry_df["soot_load_pct"] = soot_vals

    maint_df = pd.DataFrame(maint_rows)
    return telemetry_df, maint_df

def add_diff_pressure(cfg: SimConfig, telemetry_df: pd.DataFrame, vehicle_row: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 21)
    n = len(telemetry_df)

    noise = rng.normal(0, 1.0, size=n)
    diff_p = 2 + 0.02*telemetry_df["exhaust_flow_rate"] + 0.35*telemetry_df["soot_load_pct"] + noise
    diff_p += float(vehicle_row.press_sensor_bias)
    diff_p = np.clip(diff_p, 0, None)

    telemetry_df = telemetry_df.copy()
    telemetry_df["differential_pressure_kpa"] = diff_p
    return telemetry_df


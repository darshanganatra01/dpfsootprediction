import numpy as np
import pandas as pd

def build_features_from_window(window: list) -> dict:
    """
    window: list of dict raw telemetry rows (oldest -> newest)
    Returns engineered feature dict usable by model.
    """
    df = pd.DataFrame(window)

    required = [
        "vehicle_speed_kmh",
        "engine_load_pct",
        "engine_rpm",
        "ambient_temp_c",
        "exhaust_temp_pre_dpf_c",
        "exhaust_temp_post_dpf_c",
        "exhaust_flow_rate",
        "differential_pressure_kpa",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing raw telemetry column: {c}")

    df = df.copy()
    df = df.fillna(method="ffill").fillna(0)

    feat = {}

    # latest values
    feat["vehicle_speed_kmh"] = float(df["vehicle_speed_kmh"].iloc[-1])
    feat["engine_load_pct"] = float(df["engine_load_pct"].iloc[-1])
    feat["engine_rpm"] = float(df["engine_rpm"].iloc[-1])
    feat["ambient_temp_c"] = float(df["ambient_temp_c"].iloc[-1])

    feat["exhaust_temp_pre_dpf_c"] = float(df["exhaust_temp_pre_dpf_c"].iloc[-1])
    feat["exhaust_temp_post_dpf_c"] = float(df["exhaust_temp_post_dpf_c"].iloc[-1])
    feat["exhaust_flow_rate"] = float(df["exhaust_flow_rate"].iloc[-1])
    feat["differential_pressure_kpa"] = float(df["differential_pressure_kpa"].iloc[-1])

    # engineered features over window
    feat["temp_pre_roll_mean_60min"] = float(df["exhaust_temp_pre_dpf_c"].mean())
    feat["pressure_roll_mean_60min"] = float(df["differential_pressure_kpa"].mean())

    x = np.arange(len(df))
    feat["pressure_trend_60min"] = float(np.polyfit(x, df["differential_pressure_kpa"], 1)[0])
    feat["temp_trend_60min"] = float(np.polyfit(x, df["exhaust_temp_pre_dpf_c"], 1)[0])

    feat["pct_highway_60min"] = float((df["vehicle_speed_kmh"] >= 60).mean())
    feat["pct_city_60min"] = float((df["vehicle_speed_kmh"] <= 30).mean())

    # regen opportunity proxy
    feat["regen_opportunity_score"] = float(
        0.6 * feat["pct_highway_60min"] +
        0.4 * float((df["exhaust_temp_pre_dpf_c"] > 350).mean())
    )

    return feat

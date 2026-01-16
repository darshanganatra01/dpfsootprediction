import numpy as np
import pandas as pd


def _rolling_slope(x: np.ndarray) -> float:
    """
    slope of y over index 0..n-1 using least squares line fit
    """
    n = len(x)
    if n < 3:
        return np.nan
    t = np.arange(n)
    y = x.astype(float)
    t_mean = t.mean()
    y_mean = y.mean()
    denom = ((t - t_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((t - t_mean) * (y - y_mean)).sum() / denom
    return slope


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must be joined dataset (ml_base_table):
    has telemetry + trip cols + maint cols + soot_load_pct.
    """
    df = df.copy()

    # --- ensure sort ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

    # =========================
    # 1) Basic derived features
    # =========================
    df["temp_delta_dpf_c"] = df["exhaust_temp_post_dpf_c"] - df["exhaust_temp_pre_dpf_c"]

    # avoid divide by zero
    df["pressure_norm"] = df["differential_pressure_kpa"] / (df["exhaust_flow_rate"] + 1e-6)

    df["load_speed_ratio"] = df["engine_load_pct"] / (df["vehicle_speed_kmh"] + 1e-3)

    # =========================
    # 2) Rolling window features (vehicle-wise)
    # =========================
    # windows in minutes since freq=1min
    windows = {
        "10min": 10,
        "30min": 30,
        "60min": 60,
    }

    g = df.groupby("vehicle_id", group_keys=False)

    for tag, w in windows.items():
        # Rolling avg temp
        df[f"temp_pre_roll_mean_{tag}"] = g["exhaust_temp_pre_dpf_c"].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 5)).mean()
        )
        df[f"temp_pre_roll_std_{tag}"] = g["exhaust_temp_pre_dpf_c"].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 5)).std()
        )

        # Rolling avg pressure
        df[f"pressure_roll_mean_{tag}"] = g["differential_pressure_kpa"].apply(
            lambda s: s.rolling(w, min_periods=max(3, w // 5)).mean()
        )

        # pressure slope (trend)
        df[f"pressure_trend_{tag}"] = g["differential_pressure_kpa"].apply(
            lambda s: s.rolling(w, min_periods=max(5, w // 3))
            .apply(lambda x: _rolling_slope(x.values), raw=False)
        )

    # =========================
    # 3) High temperature duration (regen opportunity)
    # =========================
    HIGH_TEMP = 350.0

    def high_temp_duration(series: pd.Series) -> pd.Series:
        # consecutive minutes above threshold
        out = np.zeros(len(series), dtype=float)
        streak = 0
        for i, val in enumerate(series.values):
            if val >= HIGH_TEMP:
                streak += 1
            else:
                streak = 0
            out[i] = streak
        return pd.Series(out, index=series.index)

    df["high_temp_streak_min"] = g["exhaust_temp_pre_dpf_c"].apply(high_temp_duration)

    # =========================
    # 4) Driving mode distribution (last 60 min)
    # =========================
    # create driving mode from speed
    df["mode_idle"] = (df["vehicle_speed_kmh"] < 5).astype(int)
    df["mode_city"] = ((df["vehicle_speed_kmh"] >= 5) & (df["vehicle_speed_kmh"] < 60)).astype(int)
    df["mode_highway"] = (df["vehicle_speed_kmh"] >= 60).astype(int)

    w = 60
    df["pct_idle_60min"] = g["mode_idle"].apply(lambda s: s.rolling(w, min_periods=10).mean())
    df["pct_city_60min"] = g["mode_city"].apply(lambda s: s.rolling(w, min_periods=10).mean())
    df["pct_highway_60min"] = g["mode_highway"].apply(lambda s: s.rolling(w, min_periods=10).mean())

    # =========================
    # 5) Regen-related features
    # =========================
    # If action_type exists (from join), create specific "is_regen" and time since regen
    df["is_regen_event"] = df["action_type"].isin(["active", "passive"]).astype(int)

    # time_since_last_regen in minutes (asof join already gives maint_ts but may include inspection too)
    # We'll compute regen timestamp as last row where action_type in regen.
    def time_since_last_regen(group: pd.DataFrame) -> pd.Series:
        last_regen_ts = pd.NaT
        out = []
        for _, r in group.iterrows():
            if r["action_type"] in ("active", "passive") and pd.notna(r["maint_ts"]):
                last_regen_ts = r["maint_ts"]
            if pd.isna(last_regen_ts):
                out.append(np.nan)
            else:
                out.append((r["timestamp"] - last_regen_ts).total_seconds() / 60)
        return pd.Series(out, index=group.index)

    df["time_since_last_regen_min"] = g.apply(time_since_last_regen)

    # Regen Opportunity Score (0..1)
    # high highway pct + high temp streak => opportunity
    # This is a simple proxy, models love it.
    df["regen_opportunity_score"] = (
        0.6 * (df["pct_highway_60min"].fillna(0))
        + 0.4 * np.clip(df["high_temp_streak_min"] / 20, 0, 1)
    )

    # =========================
    # 6) Cleanup: drop helper cols
    # =========================
    df = df.drop(columns=["mode_idle", "mode_city", "mode_highway"], errors="ignore")

    return df

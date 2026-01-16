import pandas as pd

def join_tables(sensor_df: pd.DataFrame, trip_df: pd.DataFrame, maint_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime types
    sensor_df = sensor_df.copy()
    trip_df = trip_df.copy()
    maint_df = maint_df.copy()

    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    trip_df["start_time"] = pd.to_datetime(trip_df["start_time"])
    trip_df["end_time"] = pd.to_datetime(trip_df["end_time"])
    maint_df["timestamp"] = pd.to_datetime(maint_df["timestamp"])

    # 1) Join trips
    trip_cols = ["trip_id", "distance_km", "duration_min", "stop_start_count"]
    df = sensor_df.merge(trip_df[trip_cols], on="trip_id", how="left")

    # 2) ASOF join: must be sorted by merge key FIRST (timestamp), then group key
    df = df.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)

    maint = maint_df.rename(columns={"timestamp": "maint_ts"}).sort_values(["maint_ts", "vehicle_id"]).reset_index(drop=True)

    df = pd.merge_asof(
        df,
        maint,
        left_on="timestamp",
        right_on="maint_ts",
        by="vehicle_id",
        direction="backward",
        allow_exact_matches=True
    )

    df["time_since_last_maint_min"] = (df["timestamp"] - df["maint_ts"]).dt.total_seconds() / 60
    return df


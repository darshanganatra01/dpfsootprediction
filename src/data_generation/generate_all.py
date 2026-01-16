import pandas as pd
from .config import SimConfig
from .vehicles import generate_vehicles
from .trips import generate_trips
from .telemetry import generate_telemetry_for_trip
from .soot_and_maintenance import simulate_soot_and_maintenance, add_diff_pressure

def generate_all(cfg: SimConfig):
    vehicles = generate_vehicles(cfg)
    trips = generate_trips(cfg, vehicles)

    telemetry_parts = []
    maint_parts = []

    for _, trip in trips.iterrows():
        vehicle_row = vehicles.loc[vehicles.vehicle_id == trip.vehicle_id].iloc[0]

        tel = generate_telemetry_for_trip(cfg, trip, vehicle_row)
        tel, maint = simulate_soot_and_maintenance(cfg, tel, vehicle_row)
        tel = add_diff_pressure(cfg, tel, vehicle_row)

        telemetry_parts.append(tel)
        if len(maint) > 0:
            maint_parts.append(maint)

    telemetry_df = pd.concat(telemetry_parts, ignore_index=True).sort_values(["vehicle_id", "timestamp"])
    maintenance_df = (
        pd.concat(maint_parts, ignore_index=True).sort_values(["vehicle_id", "timestamp"])
        if len(maint_parts) > 0 else
        pd.DataFrame(columns=["vehicle_id", "timestamp", "action_type", "regen_success", "notes"])
    )

    return vehicles, trips, telemetry_df, maintenance_df

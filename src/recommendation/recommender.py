import pandas as pd

def recommend_action(row: pd.Series) -> dict:
    soot_pred = float(row["soot_pred"])
    opp = float(row.get("regen_opportunity_score", 0.0))
    pressure = float(row.get("differential_pressure_kpa", 0.0))

    # Thresholds
    CRITICAL_SOOT = 80
    WARNING_SOOT = 65
    HIGH_PRESSURE = 25

    LOW_SOOT = 30  # mismatch check threshold

    # ✅ 1) Mismatch case: High pressure but low soot -> inspection
    if pressure >= HIGH_PRESSURE and soot_pred < LOW_SOOT:
        return {
            "recommended_action": "INSPECTION",
            "priority": "HIGH",
            "reason": "High pressure with low soot prediction (possible ash buildup or sensor issue)"
        }

    # ✅ 2) True critical soot -> active regen
    if soot_pred >= CRITICAL_SOOT:
        return {
            "recommended_action": "ACTIVE_REGEN",
            "priority": "HIGH",
            "reason": "Predicted soot above critical threshold"
        }

    # ✅ 3) Warning soot band -> passive opportunity vs monitor
    if WARNING_SOOT <= soot_pred < CRITICAL_SOOT:
        if opp >= 0.7:
            return {
                "recommended_action": "PASSIVE_REGEN_OPPORTUNITY",
                "priority": "MEDIUM",
                "reason": "Soot rising + strong passive regen opportunity"
            }
        return {
            "recommended_action": "MONITOR",
            "priority": "MEDIUM",
            "reason": "Soot in warning range"
        }

    return {
        "recommended_action": "OK",
        "priority": "LOW",
        "reason": "Normal operation"
    }

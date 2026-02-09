import pandas as pd

def recommend_action(row: pd.Series) -> dict:
    """
    Recommendation engine based on actual model behavior.
    Thresholds calibrated to model's learned pressure-to-soot mapping.
    """
    soot_pred = float(row["soot_pred"])
    opp = float(row.get("regen_opportunity_score", 0.0))
    pressure = float(row.get("differential_pressure_kpa", 0.0))

    # ============================================
    # OPTIMAL THRESHOLDS (Based on Model Analysis)
    # ============================================
    # Your model learned:
    #  5 kPa →  14% | 10 kPa →  21% | 15 kPa →  30%
    # 20 kPa →  49% | 25 kPa →  63% | 28 kPa →  75%
    # 30 kPa →  80%
    
    INSPECTION_THRESHOLD = 78      # ~30 kPa pressure
    ACTIVE_REGEN_THRESHOLD = 65    # ~25 kPa pressure
    PASSIVE_REGEN_THRESHOLD = 50   # ~20 kPa pressure  
    MONITOR_THRESHOLD = 40         # ~18 kPa pressure
    
    HIGH_PRESSURE = 25
    LOW_SOOT = 30

    # ============================================
    # 1) ANOMALY: High pressure but low soot
    # ============================================
    if pressure >= HIGH_PRESSURE and soot_pred < LOW_SOOT:
        return {
            "recommended_action": "INSPECTION",
            "priority": "HIGH",
            "reason": "High pressure with low soot prediction (possible ash buildup or sensor issue)"
        }

    # ============================================
    # 2) CRITICAL: Soot ≥ 78% (Emergency)
    # ============================================
    if soot_pred >= INSPECTION_THRESHOLD:
        return {
            "recommended_action": "INSPECTION",
            "priority": "CRITICAL",
            "reason": f"Soot load critical at {soot_pred:.1f}% - immediate inspection required"
        }

    # ============================================
    # 3) HIGH: Soot 65-78% (Active Regen Needed)
    # ============================================
    if soot_pred >= ACTIVE_REGEN_THRESHOLD:
        return {
            "recommended_action": "ACTIVE_REGEN",
            "priority": "HIGH",
            "reason": f"Soot load at {soot_pred:.1f}% - trigger active regeneration immediately"
        }

    # ============================================
    # 4) MEDIUM-HIGH: Soot 50-65% (Passive Regen)
    # ============================================
    if soot_pred >= PASSIVE_REGEN_THRESHOLD:
        # Check if good opportunity exists
        if opp >= 0.6:
            return {
                "recommended_action": "PASSIVE_REGEN_OPPORTUNITY",
                "priority": "MEDIUM",
                "reason": f"Soot at {soot_pred:.1f}% - good conditions for passive regeneration (schedule highway driving)"
            }
        else:
            return {
                "recommended_action": "PASSIVE_REGEN_OPPORTUNITY",
                "priority": "MEDIUM",
                "reason": f"Soot at {soot_pred:.1f}% - passive regeneration recommended when opportunity arises"
            }

    # ============================================
    # 5) MEDIUM: Soot 40-50% (Monitor Closely)
    # ============================================
    if soot_pred >= MONITOR_THRESHOLD:
        return {
            "recommended_action": "MONITOR",
            "priority": "MEDIUM",
            "reason": f"Soot at {soot_pred:.1f}% - increase monitoring frequency, regeneration likely needed within 24-48 hours"
        }

    # ============================================
    # 6) NORMAL: Soot < 40% (All Good)
    # ============================================
    return {
        "recommended_action": "OK",
        "priority": "LOW",
        "reason": f"Soot at {soot_pred:.1f}% - normal operation"
    }
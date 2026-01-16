from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Minimal input format:
    This assumes you've already computed/received the SAME feature columns
    used during training OR at least the raw columns required.

    For now, we keep it flexible by accepting a dict of features.
    """
    vehicle_id: str = Field(..., example="V001")
    timestamp: str = Field(..., example="2025-11-01 08:30:00")

    # send features as dict so API stays robust
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    vehicle_id: str
    timestamp: str

    soot_pred_pct: float
    soot_pred_low_pct: float
    soot_pred_high_pct: float
    ci_level: float  # e.g. 0.90

    recommended_action: str
    priority: str
    reason: str

    drifted_features: Optional[list] = None



class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_type: str
    target: str
    n_features: int
    features: List[str]

class TelemetryRecord(BaseModel):
    timestamp: str
    vehicle_speed_kmh: float
    engine_load_pct: float
    engine_rpm: float
    ambient_temp_c: float
    exhaust_temp_pre_dpf_c: float
    exhaust_temp_post_dpf_c: float
    exhaust_flow_rate: float
    differential_pressure_kpa: float


class IngestRequest(BaseModel):
    vehicle_id: str
    record: TelemetryRecord


class PredictFromRawRequest(BaseModel):
    vehicle_id: str
    last_n: int = 60

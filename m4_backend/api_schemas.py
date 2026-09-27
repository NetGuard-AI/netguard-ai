from typing import Literal
from pydantic import BaseModel, Field

class AuthRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"

class IngestResponse(BaseModel):
    stored: bool
    flow_count: int
    state_count: int
    state_completed: bool
    forecast: dict | None = None

class PredictResponse(BaseModel):
    attack_detected: bool
    attack_probability: float = Field(ge=0, le=1)
    confidence: Literal["high","medium","low","unknown"]
    predicted_stage: str
    trend: Literal["rising","stable","falling"]

class ForecastResponse(BaseModel):
    attack_probability_timeline: list[float]
    trend: Literal["rising","stable","falling"]
    predicted_stage: str
    stage_confidence: Literal["confident","uncertain"]
    top_features: list[dict]
    explanation_text: str

class SecurityZoneResponse(BaseModel):
    zone: Literal["green","yellow","red"]
    attack_probability: float = Field(ge=0, le=1)
    reason: str

class AlertResponse(BaseModel):
    id: int
    severity: str
    message: str
    source: str
    resolved: bool
    created_at: float

class EvidenceResponse(BaseModel):
    incident_id: int
    download_format: Literal["pdf","json"]
    status: str
    path: str | None = None

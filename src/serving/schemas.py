"""
schemas.py  –  Request and response schemas for the ML serving API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── Request Schema ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:       int   = Field(..., ge=18, le=95,  example=35)
    job:       str   = Field(...,                example="management")
    marital:   str   = Field(...,                example="married")
    education: str   = Field(...,                example="tertiary")
    default:   str   = Field(...,                example="no")
    balance:   int   = Field(...,                example=1500)
    housing:   str   = Field(...,                example="yes")
    loan:      str   = Field(...,                example="no")
    contact:   str   = Field(...,                example="cellular")
    day:       int   = Field(..., ge=1,  le=31,  example=15)
    month:     str   = Field(...,                example="may")
    duration:  int   = Field(..., ge=0,          example=200)
    campaign:  int   = Field(..., ge=1,          example=2)
    pdays:     int   = Field(...,                example=-1)
    previous:  int   = Field(..., ge=0,          example=0)
    poutcome:  str   = Field(...,                example="unknown")

    @field_validator("job")
    @classmethod
    def validate_job(cls, v):
        valid = ["admin.", "unknown", "unemployed", "management", "housemaid",
                 "entrepreneur", "student", "blue-collar", "self-employed",
                 "retired", "technician", "services"]
        if v not in valid:
            raise ValueError(f"job must be one of {valid}")
        return v

    @field_validator("marital")
    @classmethod
    def validate_marital(cls, v):
        valid = ["married", "single", "divorced"]
        if v not in valid:
            raise ValueError(f"marital must be one of {valid}")
        return v

    @field_validator("default", "housing", "loan")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ["yes", "no"]:
            raise ValueError("must be 'yes' or 'no'")
        return v


# ── Response Schemas ───────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    prediction:        int   = Field(..., example=1)
    probability:       float = Field(..., example=0.82)
    threshold:         float = Field(..., example=0.5)
    decision:          str   = Field(..., example="WILL SUBSCRIBE")
    model_name:        str   = Field(..., example="improved_xgboost")
    model_version:     str   = Field(..., example="1.0")


class HealthResponse(BaseModel):
    status:            str   = Field(..., example="ok")
    model_name:        str   = Field(..., example="improved_xgboost")
    model_version:     str   = Field(..., example="1.0")
    model_path:        str   = Field(..., example="data/models/improved_model.joblib")


class BatchScoreResponse(BaseModel):
    total_records:     int   = Field(..., example=100)
    will_subscribe:    int   = Field(..., example=30)
    will_not_subscribe:int   = Field(..., example=70)
    output_path:       str   = Field(..., example="data/artifacts/batch_results.json")
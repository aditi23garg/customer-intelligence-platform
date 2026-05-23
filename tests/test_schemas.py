"""
test_schemas.py  –  Unit tests for API request/response schemas.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from pydantic import ValidationError
from src.serving.schemas import PredictRequest

VALID_PAYLOAD = {
    "age": 35, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 1500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "duration": 200, "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
}

def test_valid_payload_accepted():
    req = PredictRequest(**VALID_PAYLOAD)
    assert req.age == 35

def test_age_too_high_rejected():
    payload = {**VALID_PAYLOAD, "age": 150}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)

def test_age_too_low_rejected():
    payload = {**VALID_PAYLOAD, "age": 10}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)

def test_invalid_job_rejected():
    payload = {**VALID_PAYLOAD, "job": "astronaut"}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)

def test_invalid_marital_rejected():
    payload = {**VALID_PAYLOAD, "marital": "complicated"}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)

def test_invalid_yes_no_rejected():
    payload = {**VALID_PAYLOAD, "default": "maybe"}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)

def test_negative_duration_rejected():
    payload = {**VALID_PAYLOAD, "duration": -10}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)
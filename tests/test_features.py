"""
test_features.py  –  Unit tests for feature engineering functions.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
import numpy as np
from src.data_pipeline.features import (
    clean_data, encode_target, engineer_numeric_features,
    encode_categoricals, build_features
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_row():
    return pd.DataFrame([{
        "age": 35, "job": "management", "marital": "married",
        "education": "tertiary", "default": "no", "balance": 1500,
        "housing": "yes", "loan": "no", "contact": "cellular",
        "day": 15, "month": "may", "duration": 200, "campaign": 2,
        "pdays": -1, "previous": 0, "poutcome": "unknown", "y": "no"
    }])

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"age": 35, "job": "management", "marital": "married",
         "education": "tertiary", "default": "no", "balance": 1500,
         "housing": "yes", "loan": "no", "contact": "cellular",
         "day": 15, "month": "may", "duration": 200, "campaign": 2,
         "pdays": -1, "previous": 0, "poutcome": "unknown", "y": "no"},
        {"age": 45, "job": "technician", "marital": "single",
         "education": "secondary", "default": "no", "balance": 500,
         "housing": "no", "loan": "yes", "contact": "telephone",
         "day": 10, "month": "jun", "duration": 400, "campaign": 1,
         "pdays": 100, "previous": 2, "poutcome": "success", "y": "yes"},
    ])

# ── Tests ─────────────────────────────────────────────────────────────────────
def test_clean_data_removes_duplicates(sample_df):
    df_duped = pd.concat([sample_df, sample_df]).reset_index(drop=True)
    cleaned  = clean_data(df_duped)
    assert len(cleaned) == len(sample_df)

def test_encode_target_maps_correctly(sample_df):
    df = encode_target(sample_df.copy())
    assert set(df["y"].unique()).issubset({0, 1})
    assert df[df["y"] == 1].shape[0] == 1

def test_engineer_numeric_features_adds_columns(sample_df):
    df = engineer_numeric_features(sample_df.copy())
    for col in ["fe_never_contacted", "fe_contact_intensity",
                "fe_balance_positive", "fe_long_call", "fe_age_group"]:
        assert col in df.columns, f"Missing column: {col}"

def test_fe_never_contacted_logic(sample_df):
    df = engineer_numeric_features(sample_df.copy())
    assert df.iloc[0]["fe_never_contacted"] == 1   # pdays=-1
    assert df.iloc[1]["fe_never_contacted"] == 0   # pdays=100

def test_fe_long_call_logic(sample_df):
    df = engineer_numeric_features(sample_df.copy())
    assert df.iloc[0]["fe_long_call"] == 0   # duration=200 < 300
    assert df.iloc[1]["fe_long_call"] == 1   # duration=400 > 300

def test_encode_categoricals_fit(sample_df):
    df = encode_target(engineer_numeric_features(sample_df.copy()))
    df_enc, encoders = encode_categoricals(df.copy(), fit=True)
    assert "job" in encoders
    assert isinstance(df_enc["job"].iloc[0], (int, np.integer))

def test_encode_categoricals_serve_matches_train(sample_df):
    df = encode_target(engineer_numeric_features(sample_df.copy()))
    df_train, encoders = encode_categoricals(df.copy(), fit=True)
    df_serve, _        = encode_categoricals(df.copy(), encoders=encoders, fit=False)
    assert list(df_train["job"]) == list(df_serve["job"])

def test_build_features_no_nulls(sample_df):
    df_feat, _ = build_features(sample_df.copy(), fit=True)
    assert df_feat.isnull().sum().sum() == 0

def test_build_features_column_count(sample_df):
    df_feat, _ = build_features(sample_df.copy(), fit=True)
    assert df_feat.shape[1] == 22   # 17 original + 5 engineered
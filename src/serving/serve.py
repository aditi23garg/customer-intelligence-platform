"""
serve.py  –  FastAPI ML serving API.
Endpoints: GET /health, POST /predict, POST /batch-score
Run:  uvicorn src.serving.serve:app --reload --port 8000
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import sys
import io

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.serving.schemas import PredictRequest, PredictResponse, HealthResponse, BatchScoreResponse
from src.data_pipeline.features import build_features, load_encoders

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS = ROOT / "data" / "artifacts"
MODELS    = ROOT / "data" / "models"

# ── Load model + metadata at startup ─────────────────────────────────────────
def load_promoted_model():
    """Load whichever model the promotion gate selected."""
    gate_path = ARTIFACTS / "gate_result.json"
    if not gate_path.exists():
        raise RuntimeError("gate_result.json not found. Run train.py first.")

    with open(gate_path) as f:
        gate = json.load(f)

    promoted = gate["promoted_model"]
    meta_file = (
        ARTIFACTS / "improved_metadata.json"
        if promoted == "improved_xgboost"
        else ARTIFACTS / "baseline_metadata.json"
    )

    with open(meta_file) as f:
        metadata = json.load(f)

    model_path = (
        MODELS / "improved_model.joblib"
        if promoted == "improved_xgboost"
        else MODELS / "baseline_model.joblib"
    )
    model = joblib.load(model_path)
    encoders = load_encoders()

    return model, metadata, encoders


# Load once at startup
model, metadata, encoders = load_promoted_model()
THRESHOLD = metadata["metrics"]["threshold"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Intelligence — ML Service",
    description="Predicts campaign conversion (term deposit subscription).",
    version="1.0.0",
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status        = "ok",
        model_name    = metadata["model_name"],
        model_version = metadata["model_version"],
        model_path    = metadata["model_path"],
    )


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Convert request to dataframe
        data = request.model_dump()
        df   = pd.DataFrame([data])

        # Add dummy target for pipeline (dropped before predict)
        df["y"] = "no"

        # Apply feature engineering (serve mode — fit=False)
        df_feat, _ = build_features(df, encoders=encoders, fit=False)

        feature_cols = metadata["feature_cols"]
        X = df_feat[feature_cols].values

        prob       = float(model.predict_proba(X)[0][1])
        prediction = int(prob >= THRESHOLD)
        decision   = "WILL SUBSCRIBE" if prediction == 1 else "WILL NOT SUBSCRIBE"

        return PredictResponse(
            prediction    = prediction,
            probability   = round(prob, 4),
            threshold     = THRESHOLD,
            decision      = decision,
            model_name    = metadata["model_name"],
            model_version = metadata["model_version"],
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── /batch-score ──────────────────────────────────────────────────────────────
@app.post("/batch-score", response_model=BatchScoreResponse)
async def batch_score(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Drop target if present
        if "y" in df.columns:
            df = df.drop(columns=["y"])
        df["y"] = "no"  # dummy target

        df_feat, _ = build_features(df, encoders=encoders, fit=False)
        feature_cols = metadata["feature_cols"]
        X = df_feat[feature_cols].values

        probs       = model.predict_proba(X)[:, 1]
        predictions = (probs >= THRESHOLD).astype(int)

        results = df.drop(columns=["y"]).copy()
        results["probability"]  = np.round(probs, 4)
        results["prediction"]   = predictions
        results["decision"]     = ["WILL SUBSCRIBE" if p == 1
                                   else "WILL NOT SUBSCRIBE" for p in predictions]

        out_path = ARTIFACTS / "batch_results.json"
        results.to_json(out_path, orient="records", indent=2)

        return BatchScoreResponse(
            total_records      = len(predictions),
            will_subscribe     = int(predictions.sum()),
            will_not_subscribe = int((predictions == 0).sum()),
            output_path        = str(out_path),
        )

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=True)
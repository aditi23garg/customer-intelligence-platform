"""
train.py  –  Train baseline (Logistic Regression) and improved (XGBoost) models.
Tracks all runs with MLflow. Saves models and metadata to artifacts.
Run:  python src/training/train.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import json
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data_pipeline.features import build_features, load_encoders, get_feature_columns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
SAMPLES   = ROOT / "data" / "samples"
ARTIFACTS = ROOT / "data" / "artifacts"
MODELS    = ROOT / "data" / "models"
PLOTS     = ROOT / "data" / "plots"

for p in [ARTIFACTS, MODELS, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET         = "y"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
THRESHOLD      = 0.5

# Promotion gate margins (improved must beat baseline by these amounts)
GATE_PRAUC_MARGIN = 0.03   # PR-AUC must improve by at least 3%
GATE_F1_DROP_MAX  = 0.02   # F1 must not drop by more than 2%

mlflow.set_tracking_uri("file:///" + str(ROOT / "mlruns").replace("\\", "/"))
mlflow.set_experiment("customer-intelligence-bank-marketing")


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_prob, threshold=THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc":  round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":   round(average_precision_score(y_true, y_prob), 4),
        "f1":       round(f1_score(y_true, y_pred), 4),
        "threshold": threshold,
    }


def plot_confusion_matrix(y_true, y_prob, model_name: str, threshold=THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"]); ax.set_yticklabels(["No", "Yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    path = PLOTS / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_calibration(y_true, y_prob, model_name: str):
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, fraction_pos, "s-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    path = PLOTS / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return path


# ── Load + prepare data ───────────────────────────────────────────────────────
def load_data():
    df_raw = pd.read_csv(SAMPLES / "bank_marketing_sample.csv")
    df, encoders = build_features(df_raw, fit=True)

    # Save encoders for serve time
    from src.data_pipeline.features import save_encoders
    save_encoders(encoders)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[TARGET].values

    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y), feature_cols


# ── Train baseline ────────────────────────────────────────────────────────────
def train_baseline(X_train, X_test, y_train, y_test, feature_cols):
    print("\n" + "="*60)
    print("TRAINING: Baseline — Logistic Regression")
    print("="*60)

    params = {"C": 1.0, "max_iter": 1000, "random_state": RANDOM_STATE}

    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Plots
        cm_path   = plot_confusion_matrix(y_test, y_prob, "Baseline LR")
        cal_path  = plot_calibration(y_test, y_prob, "Baseline LR")
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(cal_path))

        run_id = run.info.run_id

    # Save model locally
    model_path = MODELS / "baseline_model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "model_name":   "baseline_logistic_regression",
        "model_version": "1.0",
        "run_id":       run_id,
        "params":       params,
        "metrics":      metrics,
        "feature_cols": feature_cols,
        "model_path":   str(model_path),
    }
    with open(ARTIFACTS / "baseline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  PR-AUC  : {metrics['pr_auc']}")
    print(f"  F1      : {metrics['f1']}")
    print(f"  Run ID  : {run_id}")
    print(f"  ✅ Baseline model saved")

    return model, metrics, metadata


# ── Train improved model ──────────────────────────────────────────────────────
def train_improved(X_train, X_test, y_train, y_test, feature_cols):
    print("\n" + "="*60)
    print("TRAINING: Improved — XGBoost")
    print("="*60)

    params = {
        "n_estimators":     200,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     RANDOM_STATE,
    }

    with mlflow.start_run(run_name="improved_xgboost") as run:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_prob)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        # Plots
        cm_path  = plot_confusion_matrix(y_test, y_prob, "Improved XGBoost")
        cal_path = plot_calibration(y_test, y_prob, "Improved XGBoost")
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(cal_path))

        run_id = run.info.run_id

    # Save model locally
    model_path = MODELS / "improved_model.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "model_name":    "improved_xgboost",
        "model_version": "1.0",
        "run_id":        run_id,
        "params":        params,
        "metrics":       metrics,
        "feature_cols":  feature_cols,
        "model_path":    str(model_path),
    }
    with open(ARTIFACTS / "improved_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  PR-AUC  : {metrics['pr_auc']}")
    print(f"  F1      : {metrics['f1']}")
    print(f"  Run ID  : {run_id}")
    print(f"  ✅ Improved model saved")

    return model, metrics, metadata


# ── Promotion gate ────────────────────────────────────────────────────────────
def promotion_gate(baseline_metrics: dict, improved_metrics: dict) -> bool:
    print("\n" + "="*60)
    print("PROMOTION GATE")
    print("="*60)

    prauc_diff = improved_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    f1_diff    = improved_metrics["f1"]     - baseline_metrics["f1"]

    print(f"  Baseline  PR-AUC: {baseline_metrics['pr_auc']}  F1: {baseline_metrics['f1']}")
    print(f"  Improved  PR-AUC: {improved_metrics['pr_auc']}  F1: {improved_metrics['f1']}")
    print(f"  PR-AUC diff : {prauc_diff:+.4f}  (need >= +{GATE_PRAUC_MARGIN})")
    print(f"  F1 diff     : {f1_diff:+.4f}  (need >= -{GATE_F1_DROP_MAX})")

    gate_passed = (
        prauc_diff >= GATE_PRAUC_MARGIN and
        f1_diff    >= -GATE_F1_DROP_MAX
    )

    if gate_passed:
        print("\n  ✅ GATE PASSED — Improved model promoted to production")
    else:
        print("\n  ❌ GATE FAILED — Baseline model remains in production")
        if prauc_diff < GATE_PRAUC_MARGIN:
            print(f"     Reason: PR-AUC improvement {prauc_diff:+.4f} < required {GATE_PRAUC_MARGIN}")
        if f1_diff < -GATE_F1_DROP_MAX:
            print(f"     Reason: F1 dropped {f1_diff:+.4f} > allowed -{GATE_F1_DROP_MAX}")

    # Save gate result
    gate_result = {
        "gate_passed":       gate_passed,
        "prauc_diff":        round(prauc_diff, 4),
        "f1_diff":           round(f1_diff, 4),
        "gate_prauc_margin": GATE_PRAUC_MARGIN,
        "gate_f1_drop_max":  GATE_F1_DROP_MAX,
        "promoted_model":    "improved_xgboost" if gate_passed else "baseline_logistic_regression",
        "baseline_metrics":  baseline_metrics,
        "improved_metrics":  improved_metrics,
    }
    with open(ARTIFACTS / "gate_result.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    return gate_passed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading and preparing data...")
    (X_train, X_test, y_train, y_test), feature_cols = load_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)} columns")

    # Train both models
    baseline_model, baseline_metrics, baseline_meta = train_baseline(
        X_train, X_test, y_train, y_test, feature_cols
    )
    improved_model, improved_metrics, improved_meta = train_improved(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # Run promotion gate
    gate_passed = promotion_gate(baseline_metrics, improved_metrics)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Promoted model: {'XGBoost' if gate_passed else 'Logistic Regression'}")
    print(f"  Artifacts saved in: {ARTIFACTS}")
    print(f"  Models saved in:    {MODELS}")
    print(f"  Plots saved in:     {PLOTS}")
    print(f"\n  Run MLflow UI with:")
    print(f"  mlflow ui --backend-store-uri {ROOT}/mlruns")
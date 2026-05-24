"""
ml_drift.py  –  Generate ML drift report using Evidently.
Simulates a data shift and compares reference vs current data.
Run:  python monitoring/ml_drift.py
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric, DataDriftTable

SAMPLES  = ROOT / "data" / "samples"
MONITOR  = ROOT / "monitoring"
REPORTS  = ROOT / "docs"
REPORTS.mkdir(exist_ok=True)


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a production data shift:
    - Age distribution shifts older
    - Balance drops (economic stress)
    - Campaign contacts increase
    - More unknown contacts
    """
    df_shifted = df.copy()
    n = len(df_shifted)

    # Shift age distribution older
    df_shifted["age"] = (df_shifted["age"] + np.random.normal(8, 3, n)).clip(18, 95).astype(int)

    # Drop balance (economic stress scenario)
    df_shifted["balance"] = (df_shifted["balance"] * 0.6 +
                              np.random.normal(-500, 200, n)).astype(int)

    # Increase campaign contacts
    df_shifted["campaign"] = (df_shifted["campaign"] +
                               np.random.randint(1, 4, n)).clip(1, 50)

    # Shift duration down (shorter calls)
    df_shifted["duration"] = (df_shifted["duration"] * 0.7).astype(int)

    print("  Drift simulation applied:")
    print(f"    Age:      {df['age'].mean():.1f} → {df_shifted['age'].mean():.1f}")
    print(f"    Balance:  {df['balance'].mean():.1f} → {df_shifted['balance'].mean():.1f}")
    print(f"    Campaign: {df['campaign'].mean():.1f} → {df_shifted['campaign'].mean():.1f}")
    print(f"    Duration: {df['duration'].mean():.1f} → {df_shifted['duration'].mean():.1f}")

    return df_shifted


def generate_drift_report(reference: pd.DataFrame,
                           current: pd.DataFrame) -> dict:
    """Generate Evidently drift report and save as HTML + JSON summary."""

    # Use only numeric columns for drift detection
    num_cols = ["age", "balance", "day", "duration", "campaign",
                "pdays", "previous"]

    ref = reference[num_cols].copy()
    cur = current[num_cols].copy()

    # Build Evidently report
    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])
    report.run(reference_data=ref, current_data=cur)

    # Save HTML report
    html_path = REPORTS / "ml_drift_report.html"
    report.save_html(str(html_path))
    print(f"  HTML report saved → {html_path}")

    # Extract summary
    result      = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    n_drifted   = result["metrics"][0]["result"]["number_of_drifted_columns"]
    n_total     = result["metrics"][0]["result"]["number_of_columns"]
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drifted_columns":        n_drifted,
        "total_columns":          n_total,
        "drift_share":            round(drift_share, 4),
        "recommendation":         "RETRAIN" if dataset_drift else "MONITOR",
        "drifted_features":       [],
    }

    # Get per-column drift info
    col_results = result["metrics"][1]["result"]["drift_by_columns"]
    for col, info in col_results.items():
        if info.get("drift_detected"):
            summary["drifted_features"].append({
                "column":    col,
                "statistic":  round(info.get("statistic", 0), 4),
                "p_value":   round(info.get("p_value", 0), 4),
            })

    json_path = REPORTS / "ml_drift_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON summary saved → {json_path}")

    return summary


if __name__ == "__main__":
    print("="*60)
    print("ML DRIFT MONITORING")
    print("="*60)

    # Load reference data (training sample)
    print("\n1. Loading reference data...")
    df_ref = pd.read_csv(SAMPLES / "bank_marketing_sample.csv")
    print(f"   Shape: {df_ref.shape}")

    # Simulate current (drifted) data
    print("\n2. Simulating production drift...")
    df_cur = simulate_drift(df_ref)

    # Generate report
    print("\n3. Generating drift report...")
    summary = generate_drift_report(df_ref, df_cur)

    print("\n" + "="*60)
    print("DRIFT SUMMARY")
    print("="*60)
    print(f"  Dataset drift detected : {summary['dataset_drift_detected']}")
    print(f"  Drifted columns        : {summary['drifted_columns']}/{summary['total_columns']}")
    print(f"  Drift share            : {summary['drift_share']}")
    print(f"  Recommendation         : {summary['recommendation']}")
    if summary["drifted_features"]:
        print(f"  Drifted features:")
        for f in summary["drifted_features"]:
            print(f"    - {f['column']}: p={f['p_value']}")
    print("\n✅ Drift report complete.")
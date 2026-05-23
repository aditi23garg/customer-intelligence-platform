"""
features.py  –  Reusable feature engineering for Bank Marketing dataset.
Same functions used at train time AND serve time (no train-serving skew).
Run directly:  python src/data_pipeline/features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"
ARTIFACTS   = ROOT / "data" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Column definitions ────────────────────────────────────────────────────────
NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

CATEGORICAL_COLS = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "poutcome"]

TARGET_COL = "y"


# ── 1. Basic cleaning ─────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicates, strip whitespace from strings.
    Does NOT touch the target column.
    """
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


# ── 2. Encode target ──────────────────────────────────────────────────────────
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert target column y: yes→1, no→0."""
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0})
    return df


# ── 3. Engineer numeric features ─────────────────────────────────────────────
def engineer_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features.
    All new columns are clearly named with fe_ prefix.
    """
    df = df.copy()

    # fe_never_contacted: 1 if this customer was never contacted before
    df["fe_never_contacted"] = (df["pdays"] == -1).astype(int)

    # fe_contact_intensity: campaign contacts normalized by previous contacts
    df["fe_contact_intensity"] = df["campaign"] / (df["previous"] + 1)

    # fe_balance_positive: 1 if account balance is positive
    df["fe_balance_positive"] = (df["balance"] > 0).astype(int)

    # fe_long_call: 1 if last call duration > 5 minutes (300 seconds)
    df["fe_long_call"] = (df["duration"] > 300).astype(int)

    # fe_age_group: bin age into groups
    df["fe_age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    return df


# ── 4. Encode categoricals ────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame,
                         encoders: dict = None,
                         fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical columns.

    At TRAIN time:   call with fit=True  → fits and returns encoders dict
    At SERVE time:   call with fit=False → uses saved encoders dict

    Returns (transformed_df, encoders_dict)
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = {
                "classes": le.classes_.tolist()
            }
        else:
            # At serve time — use saved classes
            if col not in encoders:
                raise ValueError(f"No encoder found for column: {col}")
            classes = encoders[col]["classes"]
            # Map unseen values to the last class (unknown)
            df[col] = df[col].astype(str).apply(
                lambda x: classes.index(x) if x in classes else len(classes) - 1
            )

    return df, encoders


# ── 5. Save / load encoders ───────────────────────────────────────────────────
def save_encoders(encoders: dict, path: Path = None):
    if path is None:
        path = ARTIFACTS / "encoders.json"
    with open(path, "w") as f:
        json.dump(encoders, f, indent=2)
    print(f"  Encoders saved → {path}")


def load_encoders(path: Path = None) -> dict:
    if path is None:
        path = ARTIFACTS / "encoders.json"
    with open(path) as f:
        return json.load(f)


# ── 6. Full pipeline ──────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   encoders: dict = None,
                   fit: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full feature pipeline — call this everywhere (train + serve).

    Returns (feature_df, encoders)
    feature_df has all input features + engineered features, target encoded.
    """
    df = clean_data(df)
    df = encode_target(df)
    df = engineer_numeric_features(df)
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)
    return df, encoders


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of feature columns (everything except target)."""
    return [c for c in df.columns if c != TARGET_COL]


# ── 7. Main — run as script to verify ────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("FEATURE ENGINEERING — Bank Marketing")
    print("="*60)

    # Load sample
    df_raw = pd.read_csv(SAMPLES_DIR / "bank_marketing_sample.csv")
    print(f"\n  Raw shape: {df_raw.shape}")
    print(f"  Target distribution:\n{df_raw['y'].value_counts()}")

    # Build features (fit mode)
    df_feat, encoders = build_features(df_raw, fit=True)

    print(f"\n  Feature shape: {df_feat.shape}")
    print(f"  New columns added: {[c for c in df_feat.columns if c.startswith('fe_')]}")
    print(f"  Target encoded: {df_feat['y'].value_counts().to_dict()}")
    print(f"  Any nulls: {df_feat.isnull().sum().sum()}")

    # Save encoders
    save_encoders(encoders)

    # Verify serve-time (fit=False) works with saved encoders
    loaded_encoders = load_encoders()
    df_serve = df_raw.drop(columns=[TARGET_COL])          # simulate incoming request
    df_serve[TARGET_COL] = "no"                            # dummy target
    df_serve2, _ = build_features(df_serve, encoders=loaded_encoders, fit=False)

    print(f"\n  Serve-time shape: {df_serve2.shape}")
    print(f"  ✅ Train and serve produce same columns: "
          f"{list(df_feat.columns) == list(df_serve2.columns)}")

    # Save processed sample for training step
    out_path = SAMPLES_DIR / "bank_marketing_features.csv"
    df_feat.to_csv(out_path, index=False)
    print(f"\n  Saved feature dataset → {out_path}")
    print("\n✅ Feature engineering complete.")
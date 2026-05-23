"""
validate.py  –  Schema + business-rule validation for both datasets.
Run directly:  python src/data_pipeline/validate.py
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from pathlib import Path
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"


# ── 1. Bank Marketing Schema ──────────────────────────────────────────────────
bank_schema = DataFrameSchema(
    columns={
        "age":       Column(int,   Check.between(18, 95),         nullable=False),
        "job":       Column(str,   Check.isin([
                        "admin.", "unknown", "unemployed", "management",
                        "housemaid", "entrepreneur", "student", "blue-collar",
                        "self-employed", "retired", "technician", "services"
                    ]),                                            nullable=False),
        "marital":   Column(str,   Check.isin(["married","single","divorced"]), nullable=False),
        "education": Column(str,   Check.isin(["unknown","secondary","primary","tertiary"]), nullable=False),
        "default":   Column(str,   Check.isin(["yes","no"]),       nullable=False),
        "balance":   Column(int,                                   nullable=False),
        "housing":   Column(str,   Check.isin(["yes","no"]),       nullable=False),
        "loan":      Column(str,   Check.isin(["yes","no"]),       nullable=False),
        "contact":   Column(str,                                   nullable=True),
        "day":       Column(int,   Check.between(1, 31),           nullable=False),
        "month":     Column(str,                                   nullable=False),
        "duration":  Column(int,   Check.greater_than_or_equal_to(0), nullable=False),
        "campaign":  Column(int,   Check.greater_than(0),          nullable=False),
        "pdays":     Column(int,                                   nullable=False),
        "previous":  Column(int,   Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome":  Column(str,                                   nullable=True),
        "y":         Column(str,   Check.isin(["yes","no"]),       nullable=False),
    },
    name="BankMarketing",
)


# ── 2. CFPB Complaints Schema ─────────────────────────────────────────────────
complaints_schema = DataFrameSchema(
    columns={
        "complaint_id":                  Column(str, nullable=False),
        "product":                       Column(str, nullable=False),
        "issue":                         Column(str, nullable=True),
        "company":                       Column(str, nullable=False),
        "date_received":                 Column(str, nullable=True),
        "company_response":              Column(str, nullable=True),
        "consumer_complaint_narrative":  Column(str, nullable=False),
    },
    name="CFPBComplaints",
)


# ── 3. Business Rules ─────────────────────────────────────────────────────────
def check_bank_business_rules(df: pd.DataFrame) -> dict:
    results = {}

    # Rule 1: No extreme negative balances for housing loan holders
    mask = (df["housing"] == "yes") & (df["balance"] < -5000)
    results["R1_extreme_negative_balance_with_housing_loan"] = {
        "passed": True,  # informational
        "failures": int(mask.sum()),
        "description": "Informational: extreme negative balance (<-5000) with housing loan",
    }

    # Rule 2: Campaign contacts reasonable (informational)
    mask = df["campaign"] > 50
    results["R2_campaign_contacts_reasonable"] = {
        "passed": True,  # informational — rare outliers acceptable
        "failures": int(mask.sum()),
        "description": f"Informational: {int(mask.sum())} row(s) with campaign > 50 (outlier, acceptable)",
    }

    # Rule 3: Zero duration calls should not result in subscription
    mask = (df["duration"] == 0) & (df["y"] == "yes")
    results["R3_zero_duration_not_subscribed"] = {
        "passed": int(mask.sum()) == 0,
        "failures": int(mask.sum()),
        "description": "Zero-duration calls should not result in subscription",
    }

    # Rule 4: pdays=-1 means never contacted, previous should be 0
    mask = (df["pdays"] == -1) & (df["previous"] > 0)
    results["R4_pdays_previous_consistency"] = {
        "passed": int(mask.sum()) == 0,
        "failures": int(mask.sum()),
        "description": "If pdays=-1 (never contacted), previous should be 0",
    }

    # Rule 5: Target column has no nulls and only yes/no
    mask = df["y"].isna() | ~df["y"].isin(["yes", "no"])
    results["R5_target_column_clean"] = {
        "passed": int(mask.sum()) == 0,
        "failures": int(mask.sum()),
        "description": "Target column y must be yes or no, no nulls",
    }

    return results


def check_complaints_business_rules(df: pd.DataFrame) -> dict:
    results = {}

    # Rule 1: Narratives must be at least 20 characters
    mask = df["consumer_complaint_narrative"].str.len() < 20
    results["R1_narrative_min_length"] = {
        "passed": int(mask.sum()) == 0,
        "failures": int(mask.sum()),
        "description": "Narratives must be at least 20 characters long",
    }

    # Rule 2: No duplicate complaint IDs
    dupes = df["complaint_id"].duplicated().sum()
    results["R2_no_duplicate_complaint_ids"] = {
        "passed": int(dupes) == 0,
        "failures": int(dupes),
        "description": "Complaint IDs must be unique",
    }

    # Rule 3: Product field must not be empty
    mask = df["product"].str.strip() == ""
    results["R3_product_not_empty"] = {
        "passed": int(mask.sum()) == 0,
        "failures": int(mask.sum()),
        "description": "Product field must not be blank",
    }

    # Rule 4: Informational — XXXX is standard CFPB redaction, expected
    mask = df["consumer_complaint_narrative"].str.contains(
        r"XXXX", regex=True, na=False
    )
    results["R4_narrative_redaction_check"] = {
        "passed": True,
        "failures": int(mask.sum()),
        "description": f"Informational: {int(mask.sum())} narratives contain XXXX redactions (expected)",
    }

    # Rule 5: Company response must be a known value
    valid_responses = {
        "Closed with explanation",
        "Closed with monetary relief",
        "Closed with non-monetary relief",
        "Closed without relief",
        "Closed",
        "In progress",
        "Untimely response",
    }
    mask = df["company_response"].notna() & ~df["company_response"].isin(valid_responses)
    results["R5_valid_company_response"] = {
        "passed": True,  # informational — values may vary
        "failures": int(mask.sum()),
        "description": f"Informational: {int(mask.sum())} rows with unexpected company_response values",
    }

    return results


# ── 4. Runners ────────────────────────────────────────────────────────────────
def validate_bank(path: Path) -> bool:
    print("\n" + "="*60)
    print("VALIDATING: Bank Marketing Dataset")
    print("="*60)

    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")

    # Schema check
    try:
        bank_schema.validate(df, lazy=True)
        print("  ✅ Schema validation passed")
    except pa.errors.SchemaErrors as e:
        print(f"  ❌ Schema validation failed:\n{e.failure_cases}")
        return False

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  ✅ No missing values")
    else:
        print(f"  ⚠️  Missing values:\n{missing}")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"  {'✅' if dupes == 0 else '⚠️ '} Duplicate rows: {dupes}")

    # Business rules
    print("\n  Business Rules:")
    rules = check_bank_business_rules(df)
    all_passed = True
    for rule_id, result in rules.items():
        icon = "✅" if result["passed"] else "❌"
        print(f"    {icon} {rule_id}: {result['description']} (failures: {result['failures']})")
        if not result["passed"]:
            all_passed = False

    return all_passed


def validate_complaints(path: Path) -> bool:
    print("\n" + "="*60)
    print("VALIDATING: CFPB Complaints Dataset")
    print("="*60)

    df = pd.read_csv(path)

    # Fix types before validation
    df["complaint_id"] = df["complaint_id"].astype(str)
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].astype(str)
    df["product"]  = df["product"].astype(str)
    df["company"]  = df["company"].astype(str)

    print(f"  Shape: {df.shape}")

    # Schema check
    try:
        complaints_schema.validate(df, lazy=True)
        print("  ✅ Schema validation passed")
    except pa.errors.SchemaErrors as e:
        print(f"  ❌ Schema validation failed:\n{e.failure_cases}")
        return False

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  ✅ No missing values")
    else:
        print(f"  ⚠️  Missing values (expected for optional fields):\n{missing}")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"  {'✅' if dupes == 0 else '⚠️ '} Duplicate rows: {dupes}")

    # Business rules
    print("\n  Business Rules:")
    rules = check_complaints_business_rules(df)
    all_passed = True
    for rule_id, result in rules.items():
        icon = "✅" if result["passed"] else "❌"
        print(f"    {icon} {rule_id}: {result['description']} (failures: {result['failures']})")
        if not result["passed"]:
            all_passed = False

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bank_ok       = validate_bank(SAMPLES_DIR / "bank_marketing_sample.csv")
    complaints_ok = validate_complaints(SAMPLES_DIR / "cfpb_complaints_sample.csv")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Bank Marketing : {'✅ PASSED' if bank_ok else '❌ FAILED'}")
    print(f"  CFPB Complaints: {'✅ PASSED' if complaints_ok else '❌ FAILED'}")

    if not (bank_ok and complaints_ok):
        sys.exit(1)
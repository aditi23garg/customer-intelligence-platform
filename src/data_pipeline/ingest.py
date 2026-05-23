"""
ingest.py  –  Download and save raw data samples for both ML and RAG lanes.
"""

import os
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path
import urllib3

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. UCI Bank Marketing Dataset (ML Lane) ───────────────────────────────────
def download_bank_marketing():
    """Load Bank Marketing dataset from local file."""
    print("Loading Bank Marketing dataset from local file...")

    local_path = SAMPLES_DIR / "bank-full.csv"

    if not local_path.exists():
        raise FileNotFoundError(
            f"Please download bank-full.csv manually from:\n"
            f"  https://archive.ics.uci.edu/static/public/222/bank+marketing.zip\n"
            f"and place it at: {local_path}"
        )

    # Try semicolon separator first (original UCI format), then comma
    try:
        df = pd.read_csv(local_path, sep=";")
        if df.shape[1] < 5:
            df = pd.read_csv(local_path, sep=",")
    except Exception:
        df = pd.read_csv(local_path, sep=",")

    print(f"  Loaded dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Identify target column (should be 'y' in UCI version)
    target_col = "y" if "y" in df.columns else df.columns[-1]
    print(f"  Target column: {target_col}")

    # Save a 5,000 row stratified sample
    sample = (
        df.groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), 2500), random_state=42
        ))
        .reset_index(drop=True)
        .head(5000)
    )

    out_path = SAMPLES_DIR / "bank_marketing_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"  Saved {len(sample)} rows → {out_path}")
    return out_path

# ── 2. CFPB Consumer Complaints Dataset (RAG Lane) ────────────────────────────
def download_cfpb_complaints(n_records: int = 5000):
    """Load CFPB complaints in chunks to avoid memory issues."""
    print(f"Loading CFPB Complaints in chunks...")

    local_path = SAMPLES_DIR / "complaints.csv"

    if not local_path.exists():
        raise FileNotFoundError(
            f"Place complaints.csv at: {local_path}"
        )

    cols = [
        "Complaint ID",
        "Product",
        "Issue",
        "Company",
        "Date received",
        "Company response to consumer",
        "Consumer complaint narrative",
    ]
    collected = []
    total_read = 0
    chunk_size = 10000
    print("  Reading file (this may take a moment)...")
    df = pd.read_csv(local_path, usecols=cols, chunksize=chunk_size,
                      low_memory=False, on_bad_lines="skip")

    for chunk in df:
        # Keep only rows with real narratives
        chunk = chunk[
        chunk["Consumer complaint narrative"].notna() &
        (chunk["Consumer complaint narrative"].str.strip() != "") &
        (chunk["Consumer complaint narrative"].str.strip() != "N/A") &
        (chunk["Consumer complaint narrative"].str.len() >= 20)
    ]


        collected.append(chunk)
        total_read += len(chunk)
        print(f"  ...collected {total_read} rows with narratives so far", end="\r")

        if total_read >= n_records * 3:
            break

    df = pd.concat(collected, ignore_index=True)
    print(f"\n  Total rows collected: {len(df)}")

    # Rename columns
    df = df.rename(columns={
        "Complaint ID":                   "complaint_id",
        "Product":                        "product",
        "Issue":                          "issue",
        "Company":                        "company",
        "Date received":                  "date_received",
        "Company response to consumer":   "company_response",
        "Consumer complaint narrative":   "consumer_complaint_narrative",
    })
    # Sample final n_records
    sample = df.sample(n=min(n_records, len(df)), random_state=42).reset_index(drop=True)

    out_path = SAMPLES_DIR / "cfpb_complaints_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"  Saved {len(sample)} rows → {out_path}")
    return out_path

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bank_path      = download_bank_marketing()
    complaints_path = download_cfpb_complaints()

    print("\n Ingestion complete.")
    print(f"   Bank Marketing : {bank_path}")
    print(f"   CFPB Complaints: {complaints_path}")
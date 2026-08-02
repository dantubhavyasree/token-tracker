"""
Data Collection Module - LLM Prompt Analyzer
Merges raw CSV datasets, runs feature extraction,
and saves the processed dataset to data/processed/.
"""

import os
import sys
import pandas as pd

# Allow importing from sibling folders
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from feature_engineering.extractor import extract_features_from_df

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_FILES = [
    os.path.join(RAW_DIR, "prompts_set1.csv"),
    os.path.join(RAW_DIR, "prompts_set2.csv"),
]

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features.csv")


def load_and_merge() -> pd.DataFrame:
    """Load all raw CSVs and merge into a single DataFrame."""
    dfs = []
    for path in RAW_FILES:
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}")
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()   # clean column names
        dfs.append(df)
        print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(path)}")

    if not dfs:
        raise FileNotFoundError("No raw CSV files found in data/raw/")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Prompt"])
    print(f"[INFO] Total rows after merging & deduplication: {len(merged)}")
    return merged


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 1: Load and merge raw data
    print("\n--- Step 1: Loading raw datasets ---")
    raw_df = load_and_merge()

    # Step 2: Extract features
    print("\n--- Step 2: Extracting features ---")
    features_df = extract_features_from_df(raw_df)
    print(f"[INFO] Feature matrix shape: {features_df.shape}")
    print(f"[INFO] Columns: {features_df.columns.tolist()}")

    # Step 3: Save processed dataset
    print("\n--- Step 3: Saving processed dataset ---")
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved to: {OUTPUT_FILE}")

    # Step 4: Quick summary
    print("\n--- Summary ---")
    print(features_df.describe().round(2).to_string())


if __name__ == "__main__":
    main()

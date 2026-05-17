"""
src/data_cleaning.py
====================
Load and clean raw Telco churn CSV.
Handles missing values, type coercion, and basic sanity checks.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH       = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/customers_clean.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Standardise column names ──────────────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # ── 2. Fix TotalCharges (blank strings → NaN, then float) ────────────────
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Impute with MonthlyCharges for brand-new customers (tenure == 0)
    mask_new = df["tenure"] == 0
    df.loc[mask_new, "totalcharges"] = df.loc[mask_new, "monthlycharges"]
    logger.info(f"Imputed TotalCharges for {mask_new.sum()} new customers (tenure=0)")

    # Drop any remaining nulls
    before = len(df)
    df = df.dropna(subset=["totalcharges"])
    logger.info(f"Dropped {before - len(df)} rows with null TotalCharges")

    # ── 3. Strip whitespace from object columns ───────────────────────────────
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # ── 4. Encode target variable ─────────────────────────────────────────────
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    # ── 5. Sanity checks ──────────────────────────────────────────────────────
    assert df["churn"].isna().sum() == 0,  "Nulls found in churn column"
    assert (df["tenure"] >= 0).all(),      "Negative tenure found"
    assert (df["monthlycharges"] > 0).all(), "Non-positive monthly charges found"

    logger.info(f"Clean dataset: {len(df):,} rows | Churn rate: {df['churn'].mean():.2%}")
    return df


def save(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to {path}")


if __name__ == "__main__":
    df_raw   = load_raw()
    df_clean = clean(df_raw)
    save(df_clean)
    print(df_clean.head())
    print("\nChurn distribution:\n", df_clean["churn"].value_counts())

"""
src/feature_engineering.py
==========================
Create ML-ready features from the cleaned customer dataset.
Mirrors the logic in sql/04_feature_engineering.sql.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLEAN_PATH   = Path("data/processed/customers_clean.csv")
FEATURE_PATH = Path("data/processed/churn_features.csv")

# Services used to compute total_services_count
SERVICE_COLS = [
    "onlinesecurity", "onlinebackup",
    "deviceprotection", "techsupport",
    "streamingtv", "streamingmovies",
]


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded clean data: {len(df):,} rows")
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [-1, 6, 24, 48, np.inf]
    labels = ["new", "mid", "established", "loyal"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def add_computed_features(df: pd.DataFrame) -> pd.DataFrame:
    # Average monthly bill (cross-check vs raw monthly charge)
    df["avg_monthly_bill"] = np.where(
        df["tenure"] > 0,
        (df["totalcharges"] / df["tenure"]).round(2),
        df["monthlycharges"],
    )

    # Spend intensity: how much per month relative to how new they are
    df["charge_to_tenure_ratio"] = (
        df["monthlycharges"] / (df["tenure"] + 1)
    ).round(4)

    # Month-to-month flag
    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)

    # Total active add-on services
    df["total_services_count"] = df[SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    logger.info("Added computed features: avg_monthly_bill, charge_to_tenure_ratio, "
                "is_month_to_month, total_services_count")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Binary columns
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "partner", "dependents", "phoneservice", "paperlessbilling"]:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(binary_map)

    # One-hot: Contract
    contract_dummies = pd.get_dummies(df["contract"], prefix="contract").astype(int)
    contract_dummies.columns = contract_dummies.columns.str.lower().str.replace(
        " ", "_", regex=False).str.replace("-", "_", regex=False)
    df = pd.concat([df, contract_dummies], axis=1)

    # One-hot: Internet Service
    internet_dummies = pd.get_dummies(df["internetservice"], prefix="internet").astype(int)
    internet_dummies.columns = internet_dummies.columns.str.lower().str.replace(
        " ", "_", regex=False)
    df = pd.concat([df, internet_dummies], axis=1)

    # One-hot: Payment Method
    payment_dummies = pd.get_dummies(df["paymentmethod"], prefix="payment").astype(int)
    payment_dummies.columns = payment_dummies.columns.str.lower().str.replace(
        " ", "_", regex=False).str.replace("(", "", regex=False).str.replace(
        ")", "", regex=False)
    df = pd.concat([df, payment_dummies], axis=1)

    logger.info("Encoded categoricals (binary + one-hot)")
    return df


def select_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        # Raw numericals
        "tenure", "monthlycharges", "totalcharges", "seniorcitizen",
        # Engineered
        "avg_monthly_bill", "charge_to_tenure_ratio",
        "is_month_to_month", "total_services_count",
        # Binary encoded
        "gender_enc", "partner_enc", "dependents_enc",
        "phoneservice_enc", "paperlessbilling_enc",
        # Contract one-hot (drop one to avoid multicollinearity)
        "contract_month_to_month", "contract_one_year",
        # Internet one-hot
        "internet_dsl", "internet_fiber optic",
        # Payment one-hot (drop one)
        "payment_electronic_check", "payment_mailed_check",
        "payment_bank_transfer_automatic",
    ]

    # Keep only columns that actually exist (handles variations in dummy names)
    existing = [c for c in feature_cols if c in df.columns]
    missing  = set(feature_cols) - set(existing)
    if missing:
        logger.warning(f"Some expected feature columns missing: {missing}")

    X = df[existing].fillna(0)
    y = df["churn"]
    logger.info(f"Feature matrix: {X.shape} | Target: {y.value_counts().to_dict()}")
    return X, y


def build_features(path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = load_clean(path)
    df = add_tenure_group(df)
    df = add_computed_features(df)
    df = encode_categoricals(df)
    X, y = select_model_features(df)
    return X, y


def save_features(X: pd.DataFrame, y: pd.Series, path: Path = FEATURE_PATH) -> None:
    out = X.copy()
    out["churn"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info(f"Saved feature matrix ({out.shape}) to {path}")


if __name__ == "__main__":
    X, y = build_features()
    save_features(X, y)
    print(f"\nFeatures ({X.shape[1]}):")
    print(X.dtypes)

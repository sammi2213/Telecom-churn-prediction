"""
src/predict.py
==============
Score new customers with the trained XGBoost model.
Outputs a ranked CSV of at-risk customers with SHAP-based reason codes.

Usage:
    python src/predict.py --input data/raw/new_customers.csv
    python src/predict.py --input data/raw/new_customers.csv --threshold 0.6
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from data_cleaning      import clean
from feature_engineering import (
    add_tenure_group, add_computed_features,
    encode_categoricals, select_model_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = Path("models/xgboost_churn_model.pkl")
OUTPUT_DIR  = Path("data/processed")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def score(df_raw: pd.DataFrame, model, threshold: float = 0.50) -> pd.DataFrame:
    """Full pipeline: raw → features → predictions → explanations."""

    # ── Feature engineering (same as training) ──────────────────────────────
    df = clean(df_raw) if "churn" not in df_raw.columns else df_raw.copy()
    df = add_tenure_group(df)
    df = add_computed_features(df)
    df = encode_categoricals(df)

    # We pass a dummy churn column if not present (inference mode)
    if "churn" not in df.columns:
        df["churn"] = 0

    X, _ = select_model_features(df)

    # ── Predict ──────────────────────────────────────────────────────────────
    churn_proba = model.predict_proba(X)[:, 1]
    churn_pred  = (churn_proba >= threshold).astype(int)

    # ── SHAP explanations ────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X).values  # shape (n, features)

    feature_names = list(X.columns)

    def top_reason(shap_row, n=1):
        idx = np.argsort(np.abs(shap_row))[::-1][n - 1]
        feat = feature_names[idx]
        val  = shap_row[idx]
        direction = "high risk" if val > 0 else "low risk"
        return f"{feat} ({direction}, SHAP={val:+.3f})"

    # ── Assemble output DataFrame ────────────────────────────────────────────
    result = pd.DataFrame({
        "customer_id":       df_raw.get("customerid", pd.RangeIndex(len(df))).values,
        "churn_probability": churn_proba.round(4),
        "churn_prediction":  churn_pred,
        "risk_tier":         pd.cut(
            churn_proba,
            bins=[-np.inf, 0.40, 0.65, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        ),
        "top_reason_1":  [top_reason(sv, 1) for sv in shap_values],
        "top_reason_2":  [top_reason(sv, 2) for sv in shap_values],
        "top_reason_3":  [top_reason(sv, 3) for sv in shap_values],
        "shap_values_json": [
            json.dumps(dict(zip(feature_names, sv.round(4).tolist())))
            for sv in shap_values
        ],
        "recommended_action": [
            _recommend(prob) for prob in churn_proba
        ],
    })

    result = result.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    logger.info(
        f"Scored {len(result):,} customers | "
        f"Predicted churners: {result['churn_prediction'].sum()} "
        f"({result['churn_prediction'].mean():.1%})"
    )
    return result


def _recommend(prob: float) -> str:
    if prob >= 0.75:
        return "URGENT: Call within 24h — offer 30% discount on annual plan"
    elif prob >= 0.55:
        return "Call this week — offer free upgrade or loyalty bonus"
    elif prob >= 0.40:
        return "Send personalised retention email with bundle offer"
    else:
        return "No immediate action needed — monitor next month"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score customers for churn risk")
    parser.add_argument(
        "--input", type=str,
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50,
        help="Probability threshold for churn=1 (default 0.50)",
    )
    args = parser.parse_args()

    model   = load_model()
    df_raw  = pd.read_csv(args.input)
    results = score(df_raw, model, threshold=args.threshold)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "churn_predictions.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Predictions saved to {out_path}")

    print("\n=== TOP 10 AT-RISK CUSTOMERS ===")
    print(
        results[results["churn_prediction"] == 1]
        [["customer_id", "churn_probability", "risk_tier", "top_reason_1", "recommended_action"]]
        .head(10)
        .to_string(index=False)
    )

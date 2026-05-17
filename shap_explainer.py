"""
src/shap_explainer.py
=====================
Generate SHAP (SHapley Additive exPlanations) values for the XGBoost
churn model — both global importance and per-customer explanations.

Usage:
    python src/shap_explainer.py                         # global plots
    python src/shap_explainer.py --customer_id CUST-0001 # single explanation
"""

import argparse
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = Path("models/xgboost_churn_model.pkl")
OUTPUT_DIR  = Path("reports")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {MODEL_PATH}")
    return model


def compute_shap_values(model, X: pd.DataFrame) -> shap.Explanation:
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info(f"Computed SHAP values for {len(X):,} customers")
    return shap_values


def plot_global_summary(shap_values, X: pd.DataFrame) -> None:
    """Beeswarm plot — overall feature importance."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False, plot_size=(10, 7))
    plt.title("SHAP Summary — Global Feature Importance", fontsize=14, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out}")
    plt.close()


def plot_bar_importance(shap_values, X: pd.DataFrame) -> None:
    """Mean absolute SHAP bar chart."""
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "shap_bar_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out}")
    plt.close()


def explain_customer(
    customer_id: str,
    model,
    X: pd.DataFrame,
    df_raw: pd.DataFrame,
) -> None:
    """
    Print a human-readable explanation for a single customer
    and save a waterfall plot.
    """
    if customer_id not in df_raw.index:
        # Try matching by position if ID not in index
        logger.warning(f"Customer {customer_id} not in index. Using first row.")
        idx = 0
    else:
        idx = df_raw.index.get_loc(customer_id)

    customer_X = X.iloc[[idx]]
    explainer   = shap.TreeExplainer(model)
    sv          = explainer(customer_X)

    # Churn probability
    churn_prob = model.predict_proba(customer_X)[0, 1]

    # Build a ranked DataFrame of SHAP contributions
    shap_df = (
        pd.DataFrame({
            "feature":   X.columns,
            "value":     customer_X.values[0],
            "shap":      sv.values[0],
        })
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # ── Console output ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"  CHURN EXPLANATION — Customer {customer_id}")
    print("═"*60)
    print(f"  Churn Probability : {churn_prob:.1%}")
    print(f"  Prediction        : {'⚠️  WILL CHURN' if churn_prob >= 0.5 else '✅  WILL STAY'}")
    print("─"*60)
    print(f"  {'Feature':<35} {'Value':>8}  {'SHAP Impact':>12}")
    print("─"*60)

    for _, row in shap_df.head(8).iterrows():
        direction = "▲ toward churn" if row["shap"] > 0 else "▼ away from churn"
        print(f"  {row['feature']:<35} {row['value']:>8.2f}  {row['shap']:>+8.4f}  {direction}")

    print("═"*60)

    # ── Top 3 reasons (for database / retention team) ───────────────────────
    top3 = shap_df.head(3)
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        direction = "HIGH" if row["shap"] > 0 else "LOW"
        print(f"  Reason {i}: {row['feature']} is {direction} risk signal "
              f"(SHAP={row['shap']:+.4f})")

    # ── Waterfall plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(sv[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall — Customer {customer_id} (Churn Prob: {churn_prob:.1%})",
              fontsize=12, pad=10)
    plt.tight_layout()
    out = OUTPUT_DIR / f"shap_waterfall_{customer_id.replace('/', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logger.info(f"Saved waterfall plot to {out}")
    plt.close()


def get_top_reasons(shap_row: np.ndarray, feature_names: list[str], n: int = 3) -> list[str]:
    """
    Return the top N risk factors as human-readable strings.
    Used by predict.py to populate churn_predictions table.
    """
    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feat, val in pairs[:n]:
        direction = "elevated" if val > 0 else "lower"
        reasons.append(f"{feat} → {direction} churn risk (SHAP={val:+.3f})")
    return reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability for churn model")
    parser.add_argument("--customer_id", type=str, default=None,
                        help="Specific customer ID to explain")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    X, y  = build_features()

    shap_vals = compute_shap_values(model, X)

    logger.info("Generating global SHAP plots...")
    plot_global_summary(shap_vals, X)
    plot_bar_importance(shap_vals, X)

    if args.customer_id:
        explain_customer(args.customer_id, model, X, X)
    else:
        logger.info("No --customer_id given; skipping individual explanation.")
        logger.info("Run: python src/shap_explainer.py --customer_id CUST-0001")

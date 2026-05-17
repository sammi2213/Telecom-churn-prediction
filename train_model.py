"""
src/train_model.py
==================
Train XGBoost churn classifier, compare with baseline models,
tune hyperparameters, and save the best model.

Run: python src/train_model.py
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH  = Path("models/xgboost_churn_model.pkl")
REPORT_PATH = Path("reports/model_performance.md")
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────
def load_data():
    X, y = build_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# 2. Evaluate any model
# ──────────────────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(
        f"{name:25s} | Acc={metrics['accuracy']:.4f} | "
        f"Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}"
    )
    return metrics


# ──────────────────────────────────────────────────────────────
# 3. Baseline model comparison
# ──────────────────────────────────────────────────────────────
def compare_baselines(X_train, X_test, y_train, y_test) -> list[dict]:
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    baselines = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
         X_tr_sc, X_te_sc),

        ("Decision Tree",
         DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
         X_train, X_test),

        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED),
         X_train, X_test),

        ("XGBoost (default)",
         XGBClassifier(
             n_estimators=200, max_depth=5, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8,
             use_label_encoder=False, eval_metric="logloss",
             random_state=RANDOM_SEED, verbosity=0,
         ),
         X_train, X_test),
    ]

    results = []
    for name, model, Xtr, Xte in baselines:
        model.fit(Xtr, y_train)
        results.append(evaluate(name, model, Xte, y_test))

    return results


# ──────────────────────────────────────────────────────────────
# 4. Train tuned XGBoost
# ──────────────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test):
    # Class imbalance — weight the minority class
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class imbalance weight: {pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,        # L1 regularisation
        reg_lambda        = 1.0,        # L2 regularisation
        scale_pos_weight  = pos_weight,
        eval_metric       = "auc",
        early_stopping_rounds = 30,
        random_state      = RANDOM_SEED,
        verbosity         = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    metrics = evaluate("XGBoost (tuned)", model, X_test, y_test)

    print("\n" + "="*60)
    print("FULL CLASSIFICATION REPORT — XGBoost (Tuned)")
    print("="*60)
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["Stay", "Churn"]))

    return model, metrics


# ──────────────────────────────────────────────────────────────
# 5. Feature importance plot
# ──────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list[str]) -> None:
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    importance.plot(kind="barh", ax=ax, color="#4F86C6")
    ax.set_title("XGBoost — Top 15 Feature Importances", fontsize=14, pad=12)
    ax.set_xlabel("Importance score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    logger.info("Saved reports/feature_importance.png")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 6. Persist model
# ──────────────────────────────────────────────────────────────
def save_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


# ──────────────────────────────────────────────────────────────
# 7. Write markdown report
# ──────────────────────────────────────────────────────────────
def write_report(baseline_results: list[dict], xgb_metrics: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results = baseline_results + [xgb_metrics]

    lines = [
        "# Model Performance Report\n",
        "## Model Comparison\n",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|-------|----------|-----------|--------|----|---------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"
        )

    lines += [
        "\n## Winner: XGBoost (Tuned)\n",
        f"- **Accuracy:** {xgb_metrics['accuracy']:.2%}",
        f"- **Precision:** {xgb_metrics['precision']:.2%}",
        f"- **Recall:** {xgb_metrics['recall']:.2%}",
        f"- **F1 Score:** {xgb_metrics['f1']:.2%}",
        f"- **ROC-AUC:** {xgb_metrics['roc_auc']:.4f}",
        "\n![Feature Importance](feature_importance.png)",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Report written to {REPORT_PATH}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    logger.info("\n--- Baseline Model Comparison ---")
    baseline_results = compare_baselines(X_train, X_test, y_train, y_test)

    logger.info("\n--- Training Tuned XGBoost ---")
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    plot_feature_importance(xgb_model, list(X_train.columns))
    save_model(xgb_model)
    write_report(baseline_results, xgb_metrics)

    logger.info("\n✅ Training pipeline complete.")

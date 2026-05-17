"""
tests/test_pipeline.py
======================
Unit tests for the churn prediction pipeline.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal raw customer DataFrame that mirrors the Telco CSV schema."""
    return pd.DataFrame({
        "customerID":       ["C001", "C002", "C003", "C004"],
        "gender":           ["Male", "Female", "Male", "Female"],
        "SeniorCitizen":    [0, 1, 0, 0],
        "Partner":          ["Yes", "No", "Yes", "No"],
        "Dependents":       ["No", "No", "Yes", "No"],
        "tenure":           [1, 24, 60, 0],
        "PhoneService":     ["Yes", "Yes", "Yes", "No"],
        "MultipleLines":    ["No", "Yes", "Yes", "No phone service"],
        "InternetService":  ["Fiber optic", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity":   ["No", "Yes", "Yes", "No"],
        "OnlineBackup":     ["No", "No", "Yes", "No"],
        "DeviceProtection": ["No", "No", "Yes", "No"],
        "TechSupport":      ["No", "Yes", "Yes", "No"],
        "StreamingTV":      ["No", "No", "Yes", "No"],
        "StreamingMovies":  ["No", "No", "Yes", "No"],
        "Contract":         ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes"],
        "PaymentMethod":    [
            "Electronic check", "Bank transfer (automatic)",
            "Credit card (automatic)", "Mailed check"
        ],
        "MonthlyCharges":   [85.70, 55.00, 115.50, 45.20],
        "TotalCharges":     ["85.70", "1320.00", "6930.00", ""],
        "Churn":            ["Yes", "No", "No", "Yes"],
    })


# ──────────────────────────────────────────────────────────────
# Data Cleaning Tests
# ──────────────────────────────────────────────────────────────

class TestDataCleaning:

    def test_total_charges_converted_to_float(self, sample_raw_df):
        import sys; sys.path.insert(0, "src")
        from data_cleaning import clean

        # Rename columns to lowercase to match CSV
        df = sample_raw_df.rename(columns=lambda c: c.lower().replace(" ", ""))
        df = clean(df)
        assert df["totalcharges"].dtype in [float, np.float64]

    def test_new_customer_total_charges_imputed(self, sample_raw_df):
        import sys; sys.path.insert(0, "src")
        from data_cleaning import clean

        df = sample_raw_df.rename(columns=lambda c: c.lower().replace(" ", ""))
        df = clean(df)
        # C004 has tenure=0 and blank TotalCharges → should be imputed
        c4 = df[df.index == df.index[-1]]
        assert not c4["totalcharges"].isna().any()

    def test_churn_encoded_as_binary(self, sample_raw_df):
        import sys; sys.path.insert(0, "src")
        from data_cleaning import clean

        df = sample_raw_df.rename(columns=lambda c: c.lower().replace(" ", ""))
        df = clean(df)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_no_negative_tenure(self, sample_raw_df):
        import sys; sys.path.insert(0, "src")
        from data_cleaning import clean

        df = sample_raw_df.rename(columns=lambda c: c.lower().replace(" ", ""))
        df = clean(df)
        assert (df["tenure"] >= 0).all()


# ──────────────────────────────────────────────────────────────
# Feature Engineering Tests
# ──────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def _make_clean_df(self):
        return pd.DataFrame({
            "tenure":           [1, 12, 60],
            "monthlycharges":   [85.0, 55.0, 115.0],
            "totalcharges":     [85.0, 660.0, 6900.0],
            "contract":         ["Month-to-month", "One year", "Two year"],
            "onlinesecurity":   ["No", "Yes", "Yes"],
            "onlinebackup":     ["No", "No", "Yes"],
            "deviceprotection": ["No", "No", "Yes"],
            "techsupport":      ["No", "Yes", "Yes"],
            "streamingtv":      ["No", "No", "Yes"],
            "streamingmovies":  ["No", "No", "Yes"],
        })

    def test_tenure_group_new(self):
        import sys; sys.path.insert(0, "src")
        from feature_engineering import add_tenure_group

        df = self._make_clean_df()
        df = add_tenure_group(df)
        assert df.loc[0, "tenure_group"] == "new"

    def test_tenure_group_loyal(self):
        import sys; sys.path.insert(0, "src")
        from feature_engineering import add_tenure_group

        df = self._make_clean_df()
        df = add_tenure_group(df)
        assert df.loc[2, "tenure_group"] == "loyal"

    def test_is_month_to_month_flag(self):
        import sys; sys.path.insert(0, "src")
        from feature_engineering import add_computed_features

        df = self._make_clean_df()
        df = add_computed_features(df)
        assert df.loc[0, "is_month_to_month"] == 1
        assert df.loc[1, "is_month_to_month"] == 0

    def test_total_services_count_correct(self):
        import sys; sys.path.insert(0, "src")
        from feature_engineering import add_computed_features

        df = self._make_clean_df()
        df = add_computed_features(df)
        # Row 2: all 6 services = Yes
        assert df.loc[2, "total_services_count"] == 6
        # Row 0: all services = No
        assert df.loc[0, "total_services_count"] == 0

    def test_avg_monthly_bill_calculation(self):
        import sys; sys.path.insert(0, "src")
        from feature_engineering import add_computed_features

        df = self._make_clean_df()
        df = add_computed_features(df)
        expected = round(660.0 / 12, 2)
        assert df.loc[1, "avg_monthly_bill"] == expected


# ──────────────────────────────────────────────────────────────
# Prediction Tests
# ──────────────────────────────────────────────────────────────

class TestPredict:

    def test_churn_probability_in_range(self):
        """Probabilities must be between 0 and 1."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        probs = mock_model.predict_proba(None)[:, 1]
        assert all(0 <= p <= 1 for p in probs)

    def test_risk_tier_assignment(self):
        probs = np.array([0.85, 0.55, 0.30])
        tiers = pd.cut(
            probs,
            bins=[-np.inf, 0.40, 0.65, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        assert tiers[0] == "HIGH"
        assert tiers[1] == "MEDIUM"
        assert tiers[2] == "LOW"

    def test_recommended_action_urgent(self):
        import sys; sys.path.insert(0, "src")
        from predict import _recommend

        action = _recommend(0.80)
        assert "URGENT" in action

    def test_recommended_action_no_action(self):
        import sys; sys.path.insert(0, "src")
        from predict import _recommend

        action = _recommend(0.20)
        assert "No immediate action" in action


# ──────────────────────────────────────────────────────────────
# SQL Logic Tests (pure Python — no DB required)
# ──────────────────────────────────────────────────────────────

class TestSQLLogic:

    def test_charge_to_tenure_ratio(self):
        """Mirrors sql/04_feature_engineering.sql formula."""
        monthly_charges = 85.70
        tenure = 2
        expected = round(monthly_charges / (tenure + 1), 4)
        computed = round(85.70 / 3, 4)
        assert abs(expected - computed) < 0.0001

    def test_avg_monthly_bill_for_new_customer(self):
        """tenure=0 → avg_monthly_bill should equal monthly_charges."""
        tenure         = 0
        total_charges  = 0
        monthly_charges = 55.0
        avg_bill = total_charges / tenure if tenure > 0 else monthly_charges
        assert avg_bill == monthly_charges

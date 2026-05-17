-- ============================================================
-- FILE: sql/04_feature_engineering.sql
-- PURPOSE: Transform raw customer data into ML-ready features
-- Populates the churn_features table
-- ============================================================

TRUNCATE TABLE churn_features;

INSERT INTO churn_features (
    customer_id,

    -- Encoded categoricals
    gender_encoded,
    partner_encoded,
    dependents_encoded,
    phone_service_encoded,
    paperless_billing_encoded,

    -- Contract (one-hot)
    contract_month_to_month,
    contract_one_year,
    contract_two_year,

    -- Internet service (one-hot)
    internet_dsl,
    internet_fiber,
    internet_none,

    -- Payment method (one-hot)
    payment_bank_transfer,
    payment_credit_card,
    payment_electronic_check,
    payment_mailed_check,

    -- Engineered features
    tenure_group,
    avg_monthly_bill,
    charge_to_tenure_ratio,
    is_month_to_month,
    total_services_count,

    -- Raw numericals
    tenure,
    monthly_charges,
    total_charges,
    senior_citizen,

    -- Target
    churn_encoded
)
SELECT
    c.customer_id,

    -- Binary encoding
    CASE c.gender           WHEN 'Male'   THEN 1 ELSE 0 END,
    CASE c.partner          WHEN 'Yes'    THEN 1 ELSE 0 END,
    CASE c.dependents       WHEN 'Yes'    THEN 1 ELSE 0 END,
    CASE c.phone_service    WHEN 'Yes'    THEN 1 ELSE 0 END,
    CASE c.paperless_billing WHEN 'Yes'   THEN 1 ELSE 0 END,

    -- Contract one-hot
    CASE WHEN c.contract = 'Month-to-month' THEN 1 ELSE 0 END,
    CASE WHEN c.contract = 'One year'       THEN 1 ELSE 0 END,
    CASE WHEN c.contract = 'Two year'       THEN 1 ELSE 0 END,

    -- Internet one-hot
    CASE WHEN c.internet_service = 'DSL'          THEN 1 ELSE 0 END,
    CASE WHEN c.internet_service = 'Fiber optic'  THEN 1 ELSE 0 END,
    CASE WHEN c.internet_service = 'No'           THEN 1 ELSE 0 END,

    -- Payment method one-hot
    CASE WHEN c.payment_method ILIKE '%bank transfer%'      THEN 1 ELSE 0 END,
    CASE WHEN c.payment_method ILIKE '%credit card%'        THEN 1 ELSE 0 END,
    CASE WHEN c.payment_method ILIKE '%electronic check%'   THEN 1 ELSE 0 END,
    CASE WHEN c.payment_method ILIKE '%mailed check%'       THEN 1 ELSE 0 END,

    -- Tenure group label
    CASE
        WHEN c.tenure BETWEEN 0  AND 6  THEN 'new'
        WHEN c.tenure BETWEEN 7  AND 24 THEN 'mid'
        WHEN c.tenure BETWEEN 25 AND 48 THEN 'established'
        ELSE 'loyal'
    END,

    -- Average monthly bill (Total / months, safe division)
    CASE
        WHEN c.tenure > 0 AND c.total_charges IS NOT NULL
        THEN ROUND(c.total_charges / c.tenure, 2)
        ELSE c.monthly_charges
    END,

    -- Charge-to-tenure ratio (spend intensity)
    ROUND(c.monthly_charges / (c.tenure + 1.0), 4),

    -- Is month-to-month flag
    CASE WHEN c.contract = 'Month-to-month' THEN 1 ELSE 0 END,

    -- Count of add-on services (online security, backup, etc.)
    (
        CASE WHEN c.online_security   = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN c.online_backup     = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN c.device_protection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN c.tech_support      = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN c.streaming_tv      = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN c.streaming_movies  = 'Yes' THEN 1 ELSE 0 END
    ),

    -- Raw numericals
    c.tenure,
    c.monthly_charges,
    COALESCE(c.total_charges, c.monthly_charges),   -- Impute with monthly for new customers
    c.senior_citizen,

    -- Target encoding
    CASE c.churn WHEN 'Yes' THEN 1 ELSE 0 END

FROM customers c
WHERE c.churn IS NOT NULL;    -- Only labelled records for training set

-- ============================================================
-- Validate feature engineering
-- ============================================================

-- Row count check
SELECT
    (SELECT COUNT(*) FROM customers WHERE churn IS NOT NULL) AS labelled_customers,
    (SELECT COUNT(*) FROM churn_features)                    AS feature_rows,
    (SELECT COUNT(*) FROM customers WHERE churn IS NOT NULL) =
    (SELECT COUNT(*) FROM churn_features)                    AS counts_match;

-- Churn rate preservation check
SELECT
    ROUND(AVG(churn_encoded) * 100, 2) AS churn_rate_pct_in_features
FROM churn_features;

-- Feature distribution spot check
SELECT
    ROUND(AVG(tenure), 1)                       AS avg_tenure,
    ROUND(AVG(monthly_charges), 2)              AS avg_monthly_charges,
    ROUND(AVG(avg_monthly_bill), 2)             AS avg_computed_bill,
    ROUND(AVG(charge_to_tenure_ratio), 4)       AS avg_ctr,
    ROUND(AVG(total_services_count::DECIMAL), 2) AS avg_services,
    SUM(is_month_to_month)                      AS month_to_month_count,
    SUM(contract_two_year)                      AS two_year_count
FROM churn_features;

-- ============================================================
-- Export for Python (run this to get the ML training CSV)
-- ============================================================

-- COPY (
--     SELECT * FROM churn_features ORDER BY customer_id
-- )
-- TO '/path/to/data/processed/churn_features.csv'
-- WITH (FORMAT CSV, HEADER TRUE);

SELECT 'Feature engineering complete.' AS status;

-- ============================================================
-- FILE: sql/01_create_tables.sql
-- PURPOSE: Create all tables for the Telecom Churn project
-- DATABASE: PostgreSQL 14+
-- ============================================================

-- Drop existing tables (clean slate for dev/testing)
DROP TABLE IF EXISTS churn_predictions CASCADE;
DROP TABLE IF EXISTS churn_features CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- ============================================================
-- TABLE 1: customers
-- Raw customer data as received from CRM / data warehouse
-- ============================================================
CREATE TABLE customers (
    customer_id         VARCHAR(20)     PRIMARY KEY,
    gender              VARCHAR(10),                        -- Male / Female
    senior_citizen      SMALLINT        DEFAULT 0,          -- 1 = senior, 0 = not
    partner             VARCHAR(5),                         -- Yes / No
    dependents          VARCHAR(5),                         -- Yes / No

    -- Service tenure & billing
    tenure              INT             NOT NULL,           -- Months with company
    monthly_charges     DECIMAL(8, 2)   NOT NULL,
    total_charges       DECIMAL(10, 2),                     -- Can be NULL for new customers

    -- Services subscribed
    phone_service       VARCHAR(5),
    multiple_lines      VARCHAR(20),
    internet_service    VARCHAR(20),                        -- DSL / Fiber optic / No
    online_security     VARCHAR(20),
    online_backup       VARCHAR(20),
    device_protection   VARCHAR(20),
    tech_support        VARCHAR(20),
    streaming_tv        VARCHAR(20),
    streaming_movies    VARCHAR(20),

    -- Contract & payment
    contract            VARCHAR(20)     NOT NULL,           -- Month-to-month / One year / Two year
    paperless_billing   VARCHAR(5),
    payment_method      VARCHAR(30),

    -- Target variable
    churn               VARCHAR(5),                         -- Yes / No (NULL for new/unlabelled customers)

    -- Metadata
    created_at          TIMESTAMP       DEFAULT NOW(),
    updated_at          TIMESTAMP       DEFAULT NOW()
);

-- ============================================================
-- TABLE 2: churn_features
-- Feature-engineered table used for model training & scoring
-- ============================================================
CREATE TABLE churn_features (
    customer_id                 VARCHAR(20)     PRIMARY KEY
                                                REFERENCES customers(customer_id),

    -- Encoded categoricals (for ML model consumption)
    gender_encoded              SMALLINT,       -- 0=Female, 1=Male
    partner_encoded             SMALLINT,       -- 0=No, 1=Yes
    dependents_encoded          SMALLINT,
    phone_service_encoded       SMALLINT,
    paperless_billing_encoded   SMALLINT,

    -- Contract encoding (one-hot)
    contract_month_to_month     SMALLINT,
    contract_one_year           SMALLINT,
    contract_two_year           SMALLINT,

    -- Internet service encoding (one-hot)
    internet_dsl                SMALLINT,
    internet_fiber              SMALLINT,
    internet_none               SMALLINT,

    -- Payment method encoding (one-hot)
    payment_bank_transfer       SMALLINT,
    payment_credit_card         SMALLINT,
    payment_electronic_check    SMALLINT,
    payment_mailed_check        SMALLINT,

    -- Engineered numerical features
    tenure_group                VARCHAR(10),    -- new / mid / loyal
    avg_monthly_bill            DECIMAL(8, 2),  -- total_charges / tenure
    charge_to_tenure_ratio      DECIMAL(8, 4),  -- monthly_charges / (tenure + 1)
    is_month_to_month           SMALLINT,       -- Convenience flag
    total_services_count        INT,            -- Count of active add-on services

    -- Raw numericals (scaled version stored separately if needed)
    tenure                      INT,
    monthly_charges             DECIMAL(8, 2),
    total_charges               DECIMAL(10, 2),
    senior_citizen              SMALLINT,

    -- Target (encoded)
    churn_encoded               SMALLINT,       -- 0=Stay, 1=Churn

    created_at                  TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- TABLE 3: churn_predictions
-- Model output — one row per scored customer per run
-- ============================================================
CREATE TABLE churn_predictions (
    prediction_id       SERIAL          PRIMARY KEY,
    customer_id         VARCHAR(20)     NOT NULL REFERENCES customers(customer_id),
    run_date            DATE            NOT NULL DEFAULT CURRENT_DATE,
    model_version       VARCHAR(20)     NOT NULL DEFAULT 'xgboost_v1',

    -- Prediction output
    churn_probability   DECIMAL(5, 4)   NOT NULL,   -- e.g. 0.7823
    churn_prediction    SMALLINT        NOT NULL,   -- 0 or 1
    risk_tier           VARCHAR(10),                -- HIGH / MEDIUM / LOW

    -- Top SHAP drivers (for business teams)
    top_reason_1        VARCHAR(100),
    top_reason_2        VARCHAR(100),
    top_reason_3        VARCHAR(100),
    shap_values_json    JSONB,                      -- Full SHAP vector

    -- Recommended action
    recommended_action  VARCHAR(200),

    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE (customer_id, run_date, model_version)
);

-- ============================================================
-- INDEXES for query performance
-- ============================================================
CREATE INDEX idx_customers_churn       ON customers (churn);
CREATE INDEX idx_customers_contract    ON customers (contract);
CREATE INDEX idx_customers_tenure      ON customers (tenure);
CREATE INDEX idx_predictions_run_date  ON churn_predictions (run_date);
CREATE INDEX idx_predictions_risk_tier ON churn_predictions (risk_tier);
CREATE INDEX idx_predictions_prob      ON churn_predictions (churn_probability DESC);

-- ============================================================
-- VIEWS for easy reporting
-- ============================================================

-- View: at-risk customers with contact info
CREATE OR REPLACE VIEW v_at_risk_customers AS
SELECT
    p.customer_id,
    p.churn_probability,
    p.risk_tier,
    p.top_reason_1,
    p.top_reason_2,
    p.top_reason_3,
    p.recommended_action,
    c.tenure,
    c.contract,
    c.monthly_charges,
    c.internet_service,
    c.payment_method,
    p.run_date
FROM churn_predictions p
JOIN customers c ON p.customer_id = c.customer_id
WHERE p.run_date = CURRENT_DATE
  AND p.churn_prediction = 1
ORDER BY p.churn_probability DESC;

-- View: daily summary dashboard
CREATE OR REPLACE VIEW v_daily_churn_summary AS
SELECT
    run_date,
    COUNT(*)                                                    AS total_scored,
    SUM(churn_prediction)                                       AS predicted_churners,
    ROUND(AVG(churn_probability) * 100, 2)                     AS avg_churn_prob_pct,
    SUM(CASE WHEN risk_tier = 'HIGH'   THEN 1 ELSE 0 END)     AS high_risk_count,
    SUM(CASE WHEN risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)     AS medium_risk_count,
    SUM(CASE WHEN risk_tier = 'LOW'    THEN 1 ELSE 0 END)     AS low_risk_count
FROM churn_predictions
GROUP BY run_date
ORDER BY run_date DESC;

-- Confirm creation
SELECT 'Tables and views created successfully.' AS status;

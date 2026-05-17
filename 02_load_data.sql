-- ============================================================
-- FILE: sql/02_load_data.sql
-- PURPOSE: Load raw CSV data into the customers table
-- Run AFTER 01_create_tables.sql
-- ============================================================

-- ============================================================
-- OPTION A: Load from CSV file (PostgreSQL COPY command)
-- Update the file path to match your local setup
-- ============================================================

COPY customers (
    customer_id, gender, senior_citizen, partner, dependents,
    tenure, phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges, churn
)
FROM '/path/to/your/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
WITH (
    FORMAT CSV,
    HEADER TRUE,
    NULL ''
);

-- ============================================================
-- OPTION B: Insert sample rows manually (for testing)
-- ============================================================

INSERT INTO customers (
    customer_id, gender, senior_citizen, partner, dependents,
    tenure, phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges, churn
) VALUES
-- High-risk customers (month-to-month, short tenure, high bill)
('CUST-0001', 'Male',   0, 'No',  'No',  2,  'Yes', 'No',               'Fiber optic', 'No',  'No',  'No',  'No',  'No',  'No',  'Month-to-month', 'Yes', 'Electronic check',   85.70,  171.40, 'Yes'),
('CUST-0002', 'Female', 0, 'Yes', 'No',  1,  'Yes', 'Yes',              'Fiber optic', 'No',  'No',  'No',  'No',  'Yes', 'Yes', 'Month-to-month', 'Yes', 'Electronic check',   99.65,   99.65, 'Yes'),
('CUST-0003', 'Male',   1, 'No',  'No',  3,  'Yes', 'No',               'Fiber optic', 'No',  'Yes', 'No',  'No',  'No',  'No',  'Month-to-month', 'Yes', 'Mailed check',       79.85,  239.55, 'Yes'),
('CUST-0004', 'Female', 0, 'No',  'No',  4,  'No',  'No phone service', 'DSL',         'No',  'No',  'No',  'No',  'No',  'No',  'Month-to-month', 'No',  'Electronic check',   45.20,  180.80, 'Yes'),

-- Low-risk customers (long tenure, annual/2-year contracts)
('CUST-0005', 'Male',   0, 'Yes', 'Yes', 60, 'Yes', 'Yes',              'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Two year',       'Yes', 'Credit card (auto)', 115.50, 6930.00, 'No'),
('CUST-0006', 'Female', 0, 'Yes', 'Yes', 48, 'Yes', 'No',               'DSL',         'Yes', 'Yes', 'No',  'Yes', 'No',  'No',  'One year',       'No',  'Bank transfer (auto)',55.00, 2640.00, 'No'),
('CUST-0007', 'Male',   0, 'No',  'No',  36, 'Yes', 'No',               'DSL',         'Yes', 'No',  'No',  'No',  'No',  'No',  'Two year',       'No',  'Bank transfer (auto)',49.90, 1796.40, 'No'),
('CUST-0008', 'Female', 1, 'Yes', 'No',  24, 'Yes', 'Yes',              'Fiber optic', 'No',  'Yes', 'Yes', 'No',  'Yes', 'No',  'One year',       'Yes', 'Credit card (auto)', 89.10, 2138.40, 'No'),

-- Medium-risk customers (mid tenure, mixed signals)
('CUST-0009', 'Male',   0, 'Yes', 'No',  12, 'Yes', 'No',               'DSL',         'No',  'No',  'No',  'No',  'No',  'No',  'Month-to-month', 'No',  'Mailed check',       55.90,  670.80, 'No'),
('CUST-0010', 'Female', 0, 'No',  'Yes', 8,  'Yes', 'Yes',              'Fiber optic', 'No',  'No',  'No',  'No',  'No',  'Yes', 'Month-to-month', 'Yes', 'Electronic check',   75.40,  603.20, 'Yes');

-- ============================================================
-- DATA QUALITY CHECKS after loading
-- ============================================================

-- Check total rows loaded
SELECT COUNT(*) AS total_customers FROM customers;

-- Check churn distribution
SELECT
    churn,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM customers
GROUP BY churn;

-- Check for nulls in key columns
SELECT
    COUNT(*)                                            AS total_rows,
    SUM(CASE WHEN tenure IS NULL THEN 1 ELSE 0 END)    AS null_tenure,
    SUM(CASE WHEN monthly_charges IS NULL THEN 1 ELSE 0 END) AS null_monthly_charges,
    SUM(CASE WHEN total_charges IS NULL THEN 1 ELSE 0 END)   AS null_total_charges,
    SUM(CASE WHEN contract IS NULL THEN 1 ELSE 0 END)  AS null_contract,
    SUM(CASE WHEN churn IS NULL THEN 1 ELSE 0 END)     AS null_churn
FROM customers;

-- Fix: total_charges is sometimes blank string in raw CSV — cast to NULL
UPDATE customers
SET total_charges = NULL
WHERE total_charges = 0 AND tenure = 0;

SELECT 'Data loaded and validated.' AS status;

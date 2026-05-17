-- ============================================================
-- FILE: sql/03_eda_queries.sql
-- PURPOSE: Exploratory Data Analysis queries
-- Run these to understand the dataset before modelling
-- ============================================================

-- ============================================================
-- SECTION 1: Dataset Overview
-- ============================================================

-- 1.1 Basic counts
SELECT
    COUNT(*)                                                    AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    SUM(CASE WHEN churn = 'No'  THEN 1 ELSE 0 END)            AS retained,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 1.2 Numerical feature summary statistics
SELECT
    ROUND(AVG(tenure), 2)           AS avg_tenure_months,
    MIN(tenure)                     AS min_tenure,
    MAX(tenure)                     AS max_tenure,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges,
    MIN(monthly_charges)            AS min_monthly_charges,
    MAX(monthly_charges)            AS max_monthly_charges,
    ROUND(AVG(total_charges), 2)    AS avg_total_charges
FROM customers
WHERE total_charges IS NOT NULL;

-- ============================================================
-- SECTION 2: Churn by Contract Type
-- KEY INSIGHT: Month-to-month customers churn at much higher rates
-- ============================================================
SELECT
    contract,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                 AS avg_monthly_charges
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- SECTION 3: Churn by Tenure Group
-- KEY INSIGHT: New customers (< 6 months) churn most
-- ============================================================
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 6  THEN '0-6 months (New)'
        WHEN tenure BETWEEN 7  AND 24 THEN '7-24 months (Growing)'
        WHEN tenure BETWEEN 25 AND 48 THEN '25-48 months (Established)'
        ELSE '49+ months (Loyal)'
    END AS tenure_group,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                                 AS avg_monthly_charges
FROM customers
GROUP BY 1
ORDER BY MIN(tenure);

-- ============================================================
-- SECTION 4: Churn by Internet Service Type
-- ============================================================
SELECT
    internet_service,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- SECTION 5: Churn by Monthly Charges Bucket
-- KEY INSIGHT: High bill + short tenure = very high churn
-- ============================================================
SELECT
    CASE
        WHEN monthly_charges < 35  THEN '< ₹35 (Budget)'
        WHEN monthly_charges < 65  THEN '₹35-65 (Mid)'
        WHEN monthly_charges < 85  THEN '₹65-85 (Premium)'
        ELSE '₹85+ (High-end)'
    END AS bill_bucket,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(tenure), 1)                                          AS avg_tenure
FROM customers
GROUP BY 1
ORDER BY MIN(monthly_charges);

-- ============================================================
-- SECTION 6: Churn by Payment Method
-- ============================================================
SELECT
    payment_method,
    COUNT(*)                                                        AS total,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- ============================================================
-- SECTION 7: High-Risk Segment — The "danger zone"
-- Month-to-month + Fiber + tenure < 12 months
-- ============================================================
SELECT
    COUNT(*)                                                        AS total_danger_zone,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END)                AS actually_churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE contract = 'Month-to-month'
  AND internet_service = 'Fiber optic'
  AND tenure < 12;

-- ============================================================
-- SECTION 8: Revenue at risk (monthly charges of likely churners)
-- ============================================================
SELECT
    SUM(monthly_charges)            AS total_monthly_revenue,
    SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) AS revenue_at_risk,
    ROUND(
        SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) * 100.0 /
        SUM(monthly_charges), 2
    )                               AS pct_revenue_at_risk
FROM customers;

-- ============================================================
-- SECTION 9: Senior citizen churn comparison
-- ============================================================
SELECT
    CASE senior_citizen WHEN 1 THEN 'Senior' ELSE 'Non-senior' END AS segment,
    COUNT(*)                                                        AS total,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY senior_citizen;

-- ============================================================
-- SECTION 10: Feature correlation proxy — avg charges by churn
-- ============================================================
SELECT
    churn,
    ROUND(AVG(tenure), 1)           AS avg_tenure,
    ROUND(AVG(monthly_charges), 2)  AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2)    AS avg_total_charges,
    COUNT(*)                        AS customer_count
FROM customers
WHERE total_charges IS NOT NULL
GROUP BY churn;

-- ============================================================
-- FILE: sql/05_churn_reporting.sql
-- PURPOSE: Business-facing reports after model predictions are stored
-- Run AFTER Python scoring pipeline populates churn_predictions
-- ============================================================

-- ============================================================
-- REPORT 1: Daily At-Risk Customer List (for Retention Team)
-- Ranked by churn probability — call the highest-risk first
-- ============================================================
SELECT
    ROW_NUMBER() OVER (ORDER BY p.churn_probability DESC) AS priority_rank,
    p.customer_id,
    ROUND(p.churn_probability * 100, 1)         AS churn_prob_pct,
    p.risk_tier,
    c.tenure                                    AS months_with_us,
    c.contract,
    c.monthly_charges,
    c.internet_service,
    p.top_reason_1                              AS primary_risk_factor,
    p.top_reason_2                              AS secondary_risk_factor,
    p.recommended_action
FROM churn_predictions p
JOIN customers c ON p.customer_id = c.customer_id
WHERE p.run_date = CURRENT_DATE
  AND p.churn_prediction = 1
ORDER BY p.churn_probability DESC
LIMIT 100;      -- Retention team can typically handle ~100 calls/day


-- ============================================================
-- REPORT 2: Executive Summary — Today's Risk Snapshot
-- ============================================================
SELECT
    CURRENT_DATE                                                AS report_date,
    COUNT(*)                                                    AS total_customers_scored,
    SUM(p.churn_prediction)                                     AS predicted_to_churn,
    ROUND(SUM(p.churn_prediction) * 100.0 / COUNT(*), 2)       AS predicted_churn_rate_pct,

    -- Revenue at risk
    ROUND(SUM(CASE WHEN p.churn_prediction = 1
                   THEN c.monthly_charges ELSE 0 END), 2)      AS monthly_revenue_at_risk,

    -- Risk tier breakdown
    SUM(CASE WHEN p.risk_tier = 'HIGH'   THEN 1 ELSE 0 END)   AS high_risk_customers,
    SUM(CASE WHEN p.risk_tier = 'MEDIUM' THEN 1 ELSE 0 END)   AS medium_risk_customers,
    SUM(CASE WHEN p.risk_tier = 'LOW'    THEN 1 ELSE 0 END)   AS low_risk_customers
FROM churn_predictions p
JOIN customers c ON p.customer_id = c.customer_id
WHERE p.run_date = CURRENT_DATE;


-- ============================================================
-- REPORT 3: Churn Trend Over Time (weekly rollup)
-- ============================================================
SELECT
    DATE_TRUNC('week', run_date)::DATE          AS week_starting,
    COUNT(DISTINCT run_date)                    AS days_scored,
    ROUND(AVG(daily_churners), 0)               AS avg_daily_predicted_churners,
    ROUND(AVG(daily_revenue_at_risk), 2)        AS avg_daily_revenue_at_risk
FROM (
    SELECT
        run_date,
        SUM(churn_prediction)                   AS daily_churners,
        SUM(churn_prediction * c.monthly_charges) AS daily_revenue_at_risk
    FROM churn_predictions p
    JOIN customers c ON p.customer_id = c.customer_id
    GROUP BY run_date
) daily_stats
GROUP BY 1
ORDER BY 1 DESC;


-- ============================================================
-- REPORT 4: Top Churn Drivers (what reasons come up most often)
-- ============================================================
SELECT
    top_reason_1                                AS risk_factor,
    COUNT(*)                                    AS times_cited_as_primary,
    ROUND(AVG(churn_probability) * 100, 1)      AS avg_churn_prob_when_primary
FROM churn_predictions
WHERE run_date = CURRENT_DATE
  AND churn_prediction = 1
  AND top_reason_1 IS NOT NULL
GROUP BY top_reason_1
ORDER BY times_cited_as_primary DESC;


-- ============================================================
-- REPORT 5: Segment-Level Churn Rate (for product team)
-- ============================================================
SELECT
    c.contract,
    c.internet_service,
    CASE
        WHEN c.tenure < 6  THEN 'New (< 6 mo)'
        WHEN c.tenure < 24 THEN 'Mid (6-24 mo)'
        ELSE 'Loyal (24+ mo)'
    END                                         AS tenure_band,
    COUNT(*)                                    AS customer_count,
    SUM(p.churn_prediction)                     AS predicted_churners,
    ROUND(SUM(p.churn_prediction) * 100.0 / COUNT(*), 2) AS predicted_churn_rate_pct,
    ROUND(AVG(c.monthly_charges), 2)            AS avg_monthly_charges
FROM churn_predictions p
JOIN customers c ON p.customer_id = c.customer_id
WHERE p.run_date = CURRENT_DATE
GROUP BY c.contract, c.internet_service, 3
HAVING COUNT(*) >= 10   -- Only show segments with meaningful sample size
ORDER BY predicted_churn_rate_pct DESC;


-- ============================================================
-- REPORT 6: Retention ROI Estimator
-- If we retain X% of high-risk customers with a Y% discount offer
-- ============================================================
WITH high_risk AS (
    SELECT
        p.customer_id,
        c.monthly_charges,
        p.churn_probability
    FROM churn_predictions p
    JOIN customers c ON p.customer_id = c.customer_id
    WHERE p.run_date = CURRENT_DATE
      AND p.risk_tier = 'HIGH'
)
SELECT
    COUNT(*)                                        AS high_risk_customers,
    ROUND(SUM(monthly_charges), 2)                 AS monthly_revenue_at_risk,
    -- Assume 30% of called customers accept the offer and stay
    ROUND(SUM(monthly_charges) * 0.30, 2)          AS retained_revenue_30pct_success,
    -- Net after 20% discount on retained customers
    ROUND(SUM(monthly_charges) * 0.30 * 0.80, 2)  AS net_retained_revenue_after_20pct_discount,
    -- Annual projection
    ROUND(SUM(monthly_charges) * 0.30 * 0.80 * 12, 2) AS annual_net_retention_value
FROM high_risk;


-- ============================================================
-- REPORT 7: Model Performance Tracking (actual vs predicted)
-- Run this after 30 days to validate model accuracy
-- ============================================================
SELECT
    p.run_date,
    p.model_version,
    COUNT(*)                                                AS total_predictions,
    -- True positives: predicted churn=1 and actually churned
    SUM(CASE WHEN p.churn_prediction = 1 AND c.churn = 'Yes' THEN 1 ELSE 0 END) AS true_positives,
    -- False positives: predicted churn=1 but stayed
    SUM(CASE WHEN p.churn_prediction = 1 AND c.churn = 'No'  THEN 1 ELSE 0 END) AS false_positives,
    -- True negatives: predicted churn=0 and stayed
    SUM(CASE WHEN p.churn_prediction = 0 AND c.churn = 'No'  THEN 1 ELSE 0 END) AS true_negatives,
    -- False negatives: predicted stay but churned
    SUM(CASE WHEN p.churn_prediction = 0 AND c.churn = 'Yes' THEN 1 ELSE 0 END) AS false_negatives,
    -- Derived metrics
    ROUND(
        SUM(CASE WHEN p.churn_prediction = c.churn::VARCHAR = 'Yes'::VARCHAR THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                       AS accuracy_pct
FROM churn_predictions p
JOIN customers c ON p.customer_id = c.customer_id
WHERE c.churn IS NOT NULL   -- Only rows with known outcomes
GROUP BY p.run_date, p.model_version
ORDER BY p.run_date DESC;

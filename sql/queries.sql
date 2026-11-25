-- Q1: Revenue by plan, successful recharges only
SELECT
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue_inr,
    ROUND(AVG(r.amount_inr), 2) AS average_recharge_inr,
    COUNT(DISTINCT r.subscriber_id) AS distinct_subscribers
FROM recharges r
INNER JOIN subscribers s ON r.subscriber_id = s.subscriber_id
INNER JOIN plans p ON s.plan_id = p.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue_inr DESC;

-- Q2: Subscribers with no successful recharge
SELECT
    s.subscriber_id,
    s.tenure_months
FROM subscribers s
LEFT JOIN recharges r
    ON s.subscriber_id = r.subscriber_id
   AND r.status = 'success'
WHERE r.recharge_id IS NULL
ORDER BY s.subscriber_id;

-- Q3: Rank, dense rank and row number within each city
WITH subscriber_spend AS (
    SELECT
        s.subscriber_id,
        s.city,
        SUM(r.amount_inr) AS total_spend
    FROM subscribers s
    INNER JOIN recharges r
        ON s.subscriber_id = r.subscriber_id
    WHERE r.status = 'success'
    GROUP BY s.subscriber_id, s.city
)
SELECT
    city,
    subscriber_id,
    total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS rank_value,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank_value,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC, subscriber_id
    ) AS row_number_value
FROM subscriber_spend
ORDER BY city, total_spend DESC, subscriber_id;

-- Q4: Running total of successful recharges for TP00010
SELECT
    recharge_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (
        ORDER BY recharge_date, recharge_id
    ) AS running_total_inr
FROM recharges
WHERE subscriber_id = 'TP00010'
  AND status = 'success'
ORDER BY recharge_date, recharge_id;

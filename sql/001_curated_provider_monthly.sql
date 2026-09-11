-- Curated aggregate: provider billing behavior by month.
-- Portable to Snowflake as-is (ANSI SQL + DATE_TRUNC, standard in Snowflake).
-- In production this would be materialized as a Snowflake table/dynamic table
-- refreshed on a schedule (e.g. `CREATE DYNAMIC TABLE ... TARGET_LAG = '1 hour'`).

SELECT
    p.PROVIDER_ID,
    p.PROVIDER_NAME,
    p.SPECIALTY,
    p.STATE,
    DATE_TRUNC('month', c.SERVICE_DATE)   AS SERVICE_MONTH,
    COUNT(*)                               AS CLAIM_COUNT,
    SUM(c.BILLED_AMOUNT)                   AS TOTAL_BILLED,
    SUM(c.PAID_AMOUNT)                     AS TOTAL_PAID,
    AVG(c.BILLED_AMOUNT)                   AS AVG_BILLED,
    SUM(CASE WHEN c.CLAIM_STATUS = 'Denied' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(COUNT(*), 0)              AS DENIAL_RATE
FROM claims c
JOIN providers p ON c.PROVIDER_ID = p.PROVIDER_ID
GROUP BY 1, 2, 3, 4, 5
ORDER BY SERVICE_MONTH, TOTAL_BILLED DESC;

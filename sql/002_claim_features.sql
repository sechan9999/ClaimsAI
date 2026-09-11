-- Per-claim feature engineering for fraud/anomaly detection.
-- Pure SQL window functions -- portable to Snowflake unchanged. Computing
-- these in-warehouse (rather than pulling all raw claims into Python) is
-- what makes this scale past a laptop-sized sample: only the feature table
-- below needs to leave the warehouse.
--
-- Features:
--   amount_z_in_peer_group : how far this claim's billed amount is from the
--                             mean for its own procedure code, in std devs.
--                             Upcoding/outlier claims show up here.
--   patient_provider_daily_count : how many claims this patient+provider pair
--                             filed on the same service date (phantom/duplicate
--                             billing signal).
--   patient_provider_30d_count : rolling 30-day claim count for the same
--                             patient+provider pair (excess-frequency signal).
--   submission_lag_days   : days between service and submission (very long
--                             or negative lags are a data-quality/fraud flag).

WITH peer_stats AS (
    SELECT
        PROCEDURE_CODE,
        AVG(BILLED_AMOUNT) AS PEER_MEAN,
        STDDEV_POP(BILLED_AMOUNT) AS PEER_STDDEV
    FROM claims
    GROUP BY PROCEDURE_CODE
),
same_day AS (
    SELECT
        PATIENT_ID,
        PROVIDER_ID,
        SERVICE_DATE,
        COUNT(*) AS PATIENT_PROVIDER_DAILY_COUNT
    FROM claims
    GROUP BY PATIENT_ID, PROVIDER_ID, SERVICE_DATE
),
rolling AS (
    SELECT
        CLAIM_ID,
        COUNT(*) OVER (
            PARTITION BY PATIENT_ID, PROVIDER_ID
            ORDER BY EPOCH(SERVICE_DATE)
            RANGE BETWEEN 30 * 86400 PRECEDING AND CURRENT ROW
        ) AS PATIENT_PROVIDER_30D_COUNT
    FROM claims
)
SELECT
    c.CLAIM_ID,
    c.PATIENT_ID,
    c.PROVIDER_ID,
    c.PROCEDURE_CODE,
    c.BILLED_AMOUNT,
    c.SERVICE_DATE,
    c.SUBMISSION_DATE,
    DATE_DIFF('day', c.SERVICE_DATE, c.SUBMISSION_DATE) AS SUBMISSION_LAG_DAYS,
    (c.BILLED_AMOUNT - ps.PEER_MEAN) / NULLIF(ps.PEER_STDDEV, 0) AS AMOUNT_Z_IN_PEER_GROUP,
    sd.PATIENT_PROVIDER_DAILY_COUNT,
    r.PATIENT_PROVIDER_30D_COUNT
FROM claims c
JOIN peer_stats ps ON c.PROCEDURE_CODE = ps.PROCEDURE_CODE
JOIN same_day sd
    ON c.PATIENT_ID = sd.PATIENT_ID
   AND c.PROVIDER_ID = sd.PROVIDER_ID
   AND c.SERVICE_DATE = sd.SERVICE_DATE
JOIN rolling r ON c.CLAIM_ID = r.CLAIM_ID;

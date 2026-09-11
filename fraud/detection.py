"""
Fraud / anomaly detection over patient claims.

Pipeline:
  1. Pull per-claim features computed in-warehouse (sql/002_claim_features.sql)
     -- peer-group amount z-score, same-day claim count, rolling 30-day
     claim count, submission lag.
  2. Score every claim with an IsolationForest over those features (unsupervised
     -- no fraud labels are used, matching how this would run in production
     where confirmed fraud labels are scarce and delayed).
  3. Turn the numeric features into a 0-100 risk score and a plain-language
     explanation per claim, built directly from the feature values (this is
     the "always available" explanation path -- reporting/summarize.py can
     additionally hand the same features to an LLM provider for a more
     fluent narrative when one is configured).

This mirrors the RxHCC fraud detection app's rule-engine + explainability
approach, but grounded in claims aggregated straight out of the warehouse.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from warehouse.connection import run_sql_file, run_query

_FEATURES_SQL = os.path.join(os.path.dirname(__file__), "..", "sql", "002_claim_features.sql")

_MODEL_FEATURES = [
    "AMOUNT_Z_IN_PEER_GROUP",
    "PATIENT_PROVIDER_DAILY_COUNT",
    "PATIENT_PROVIDER_30D_COUNT",
    "SUBMISSION_LAG_DAYS",
]


def _load_features() -> pd.DataFrame:
    df = run_sql_file(_FEATURES_SQL)
    df["AMOUNT_Z_IN_PEER_GROUP"] = df["AMOUNT_Z_IN_PEER_GROUP"].fillna(0.0)
    return df


def _explain(row: pd.Series) -> list[str]:
    reasons = []
    z = row["AMOUNT_Z_IN_PEER_GROUP"]
    if z >= 3:
        reasons.append(
            f"Billed ${row['BILLED_AMOUNT']:,.0f} for {row['PROCEDURE_CODE']} is "
            f"{z:.1f} standard deviations above the typical amount for that procedure."
        )
    if row["PATIENT_PROVIDER_DAILY_COUNT"] > 1:
        reasons.append(
            f"{int(row['PATIENT_PROVIDER_DAILY_COUNT'])} claims were filed for this "
            f"patient by this provider on the same service date (possible duplicate/phantom billing)."
        )
    if row["PATIENT_PROVIDER_30D_COUNT"] > 5:
        reasons.append(
            f"{int(row['PATIENT_PROVIDER_30D_COUNT'])} claims for this patient-provider pair "
            f"within a rolling 30-day window is well above typical visit frequency."
        )
    lag = row["SUBMISSION_LAG_DAYS"]
    if lag > 60 or lag < 0:
        reasons.append(f"Submission lag of {int(lag)} days between service and filing is atypical.")
    if not reasons:
        reasons.append("No single feature stands out; flagged on combined anomaly score.")
    return reasons


def score_claims(top_n: int | None = None) -> pd.DataFrame:
    """
    Returns claims ranked by risk score (0-100, higher = more anomalous),
    with a RISK_TIER and human-readable EXPLANATION for each.
    """
    features = _load_features()
    X = features[_MODEL_FEATURES].to_numpy()

    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42
    )
    model.fit(X)
    # decision_function: higher = more normal. Flip and rescale to 0-100.
    raw = -model.decision_function(X)
    risk_score = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    features["RISK_SCORE"] = risk_score.round(1)
    features["RISK_TIER"] = pd.cut(
        features["RISK_SCORE"], bins=[-1, 50, 80, 101], labels=["Low", "Medium", "High"]
    )
    features["EXPLANATION"] = features.apply(lambda r: " ".join(_explain(r)), axis=1)

    out = features.sort_values("RISK_SCORE", ascending=False)
    if top_n:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def claim_detail(claim_id: str) -> dict:
    """Full context for one claim: raw claim row + its risk features/explanation."""
    scored = score_claims()
    row = scored[scored["CLAIM_ID"] == claim_id]
    if row.empty:
        raise KeyError(f"Unknown claim_id: {claim_id}")
    row = row.iloc[0].to_dict()

    raw_claim = run_query(
        f"""
        SELECT c.*, p.PROVIDER_NAME, p.SPECIALTY
        FROM claims c
        JOIN providers p ON c.PROVIDER_ID = p.PROVIDER_ID
        WHERE c.CLAIM_ID = '{claim_id}'
        """
    )
    return {"features": row, "claim": raw_claim.iloc[0].to_dict() if not raw_claim.empty else {}}

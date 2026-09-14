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
from fraud.benford import provider_benford_scores

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
    if row.get("BENFORD_FLAG"):
        reasons.append(
            f"This claim's provider ({row['PROVIDER_ID']}) bills amounts whose leading-digit "
            f"pattern deviates significantly from Benford's Law (chi²={row['BENFORD_CHI2']:.1f}, "
            f"p={row['BENFORD_P_VALUE']:.4f} across {int(row['BENFORD_N_CLAIMS'])} claims) -- a "
            f"signal of statistically manufactured rather than naturally varied billing, independent "
            f"of any single claim's amount."
        )
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


def _attach_benford(features: pd.DataFrame) -> pd.DataFrame:
    """
    Join provider-level Benford's Law results onto per-claim rows. This is
    a group-level signal (see fraud/benford.py) -- every claim from a
    flagged provider carries the same BENFORD_* values, distinct from the
    per-claim IsolationForest features which vary claim to claim.
    Providers with too few claims to test (see MIN_CLAIMS_FOR_TEST) simply
    get no Benford columns and are never flagged on this signal.
    """
    benford = provider_benford_scores().rename(
        columns={
            "CHI2_STATISTIC": "BENFORD_CHI2",
            "P_VALUE": "BENFORD_P_VALUE",
            "N_CLAIMS": "BENFORD_N_CLAIMS",
        }
    )
    merged = features.merge(benford, on="PROVIDER_ID", how="left")
    merged["BENFORD_FLAG"] = merged["BENFORD_FLAG"].fillna(False)
    return merged


def score_claims(top_n: int | None = None) -> pd.DataFrame:
    """
    Returns claims ranked by risk score (0-100, higher = more anomalous),
    with a RISK_TIER and human-readable EXPLANATION for each.
    """
    features = _load_features()
    features = _attach_benford(features)
    X = features[_MODEL_FEATURES].to_numpy()

    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42
    )
    model.fit(X)
    # decision_function: higher = more normal. Flip and rescale to 0-100.
    raw = -model.decision_function(X)
    risk_score = 100 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    features["RISK_SCORE"] = risk_score.round(1)

    # Deliberately NOT blended into RISK_SCORE. An earlier version floored
    # every claim from a Benford-flagged provider at a minimum score, but
    # validation (fraud/validate_against_labels.py) showed this measurably
    # hurt the per-claim triage ranking: a flagged provider can carry
    # hundreds of otherwise-ordinary claims, and forcing all of them above
    # the Medium threshold pushed genuinely higher-risk claims out of the
    # top-N a triage queue would actually work through. Benford's Law is a
    # provider-level statement ("this billing pattern looks manufactured"),
    # not a claim-level one, so it's surfaced as its own signal -- in
    # EXPLANATION text here, and as a standalone provider table in the UI --
    # rather than collapsed into the same per-claim ranking.
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
        """
        SELECT c.*, p.PROVIDER_NAME, p.SPECIALTY
        FROM claims c
        JOIN providers p ON c.PROVIDER_ID = p.PROVIDER_ID
        WHERE c.CLAIM_ID = ?
        """,
        params=[claim_id],
    )
    return {"features": row, "claim": raw_claim.iloc[0].to_dict() if not raw_claim.empty else {}}

"""
Dev/QA-only script: validates the anomaly detector against the synthetic
ground-truth fraud labels generated alongside the data (claims_fraud_labels.parquet).

This file is NOT part of the production app -- a real deployment wouldn't
have ground truth to check against at scoring time. It exists purely to
confirm the unsupervised model (fraud/detection.py) actually surfaces the
injected fraud patterns before shipping it, and to report precision/recall
at a few top-N cutoffs a fraud analyst might realistically review.
"""
import os
import pandas as pd
from fraud.detection import score_claims
from fraud.benford import provider_benford_scores, FLAG_P_VALUE

LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "claims_fraud_labels.parquet")
CLAIMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "claims.parquet")


def main():
    labels = pd.read_parquet(LABELS_PATH)
    scored = score_claims()
    merged = scored.merge(labels, on="CLAIM_ID", how="left")

    total_fraud = merged["IS_FRAUD"].sum()
    print(f"Total claims: {len(merged):,} | actual fraud: {total_fraud:,} "
          f"({total_fraud / len(merged) * 100:.2f}%)\n")

    for top_n in [500, 1000, 2000, 3000]:
        top = merged.head(top_n)
        caught = top["IS_FRAUD"].sum()
        precision = caught / top_n
        recall = caught / total_fraud
        print(f"Top {top_n:>5}: precision={precision:.1%}  recall={recall:.1%}  "
              f"(caught {caught:,} of {total_fraud:,} known fraud claims)")

    print("\nFraud-type capture rate in top 3,000 by risk score:")
    top3000_ids = set(merged.head(3000)["CLAIM_ID"])
    by_type = labels[labels["IS_FRAUD"]].groupby("FRAUD_TYPE").apply(
        lambda g: (g["CLAIM_ID"].isin(top3000_ids).sum(), len(g))
    )
    for fraud_type, (caught, total) in by_type.items():
        print(f"  {fraud_type:<20} {caught:>5}/{total:<5} ({caught/total:.1%})")

    print("\n--- Provider-level Benford's Law check (group signal, separate from the per-claim model above) ---")
    claims = pd.read_parquet(CLAIMS_PATH)
    fraud_claims = claims.merge(labels[labels["IS_FRAUD"]], on="CLAIM_ID")
    actual_fraud_providers = set(fraud_claims["PROVIDER_ID"].unique())

    benford = provider_benford_scores()
    flagged_providers = set(benford[benford["BENFORD_FLAG"]]["PROVIDER_ID"])
    true_positives = flagged_providers & actual_fraud_providers

    precision = len(true_positives) / len(flagged_providers) if flagged_providers else 0.0
    recall = len(true_positives) / len(actual_fraud_providers) if actual_fraud_providers else 0.0
    print(f"Providers tested: {len(benford):,} | providers carrying >=1 injected-fraud claim: "
          f"{len(actual_fraud_providers):,}")
    print(f"Benford-flagged providers (p<{FLAG_P_VALUE}): {len(flagged_providers):,} | "
          f"of which actually fraud-carrying: {len(true_positives):,}")
    print(f"Provider-level precision={precision:.1%}  recall={recall:.1%}")
    print(
        "This is a *group* signal over a whole provider's billing pattern, not a per-claim one -- "
        "expect it to catch providers whose fraud is spread across many claims (e.g. systematic "
        "upcoding/outlier amounts) rather than a single anomalous claim from an otherwise normal "
        "provider, and expect some false positives from providers whose legitimate billing happens "
        "to cluster (e.g. a narrow-specialty practice with few distinct price points)."
    )


if __name__ == "__main__":
    main()

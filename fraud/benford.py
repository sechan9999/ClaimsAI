"""
Benford's Law analysis over provider billing amounts.

Benford's Law: for many kinds of naturally-occurring numeric data, the
leading (first significant) digit d in {1..9} follows P(d) = log10(1 + 1/d)
-- not a uniform 1/9. Amounts that are typed, rounded, capped, or otherwise
manufactured tend to deviate from this distribution. It's a *group-level*
statistical test, not a per-claim feature: you need a reasonable number of
billed amounts from one source before "the distribution of their leading
digits" is even a meaningful question.

That makes it a natural complement to detection.py's IsolationForest, which
scores individual claims as outliers but has nothing to say about a
provider whose amounts are each unremarkable on their own yet collectively
look statistically manufactured (round-number billing, price-capping just
under an approval threshold, copy-pasted amounts across claims). This mirrors
the NEMESIS grant-monitoring system's fraud approach -- Benford's Law with
chi-square testing -- applied here to provider billing instead of grant
amounts.

Output is a provider-level score; fraud/detection.py joins it onto
per-claim rows by PROVIDER_ID so a claim's explanation can note when its
provider's billing pattern is itself flagged, alongside the per-claim
anomaly reasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from warehouse.connection import run_query

# Benford's Law needs enough observations per provider for the chi-square
# test to be meaningful -- below this, a p-value is just noise, so those
# providers are left unscored rather than reported with false confidence.
MIN_CLAIMS_FOR_TEST = 30

# Significance threshold for flagging a provider. Deliberately conservative
# (not the common 0.05) because this runs across every provider at once --
# testing many providers means some will cross 0.05 by chance alone.
FLAG_P_VALUE = 0.01

_DIGITS = np.arange(1, 10)
_BENFORD_EXPECTED_PROPORTIONS = np.log10(1 + 1 / _DIGITS)


def _leading_digit(amount: float) -> int:
    """First non-zero digit of a positive amount, e.g. 245.90 -> 2, 0.087 -> 8."""
    s = f"{abs(float(amount)):.10f}".replace(".", "")
    for ch in s:
        if ch != "0":
            return int(ch)
    return 0  # only reachable for an amount of exactly 0


def provider_benford_scores() -> pd.DataFrame:
    """
    One row per provider with >= MIN_CLAIMS_FOR_TEST claims:
      N_CLAIMS         claims used in the test
      CHI2_STATISTIC    chi-square goodness-of-fit statistic vs. Benford's
                        expected leading-digit distribution (higher = further
                        from natural)
      P_VALUE           probability of this much deviation occurring by
                        chance if the amounts were Benford-distributed
      BENFORD_FLAG      True when P_VALUE < FLAG_P_VALUE -- this provider's
                        billing amounts, as a group, look statistically
                        manufactured rather than naturally distributed
    """
    claims = run_query("SELECT PROVIDER_ID, BILLED_AMOUNT FROM claims")
    claims["LEADING_DIGIT"] = claims["BILLED_AMOUNT"].apply(_leading_digit)
    claims = claims[claims["LEADING_DIGIT"] > 0]  # exclude zero/degenerate amounts

    rows = []
    for provider_id, grp in claims.groupby("PROVIDER_ID"):
        n = len(grp)
        if n < MIN_CLAIMS_FOR_TEST:
            continue
        observed = grp["LEADING_DIGIT"].value_counts().reindex(_DIGITS, fill_value=0).to_numpy()
        expected = _BENFORD_EXPECTED_PROPORTIONS * n
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        rows.append(
            {
                "PROVIDER_ID": provider_id,
                "N_CLAIMS": n,
                "CHI2_STATISTIC": round(float(chi2), 2),
                "P_VALUE": round(float(p_value), 4),
                "BENFORD_FLAG": bool(p_value < FLAG_P_VALUE),
            }
        )

    out = pd.DataFrame(rows, columns=["PROVIDER_ID", "N_CLAIMS", "CHI2_STATISTIC", "P_VALUE", "BENFORD_FLAG"])
    return out.sort_values("CHI2_STATISTIC", ascending=False).reset_index(drop=True)

"""
Narrative summarization over query results and portfolio-level trends.

Two paths, mirroring llm/provider.py's graceful degradation:
  - Live Cortex/LLM provider available -> hand it a compact table + the
    computed stats and let it write the narrative.
  - No provider available (local demo)  -> compose the narrative directly
    from the computed stats with plain-language templates. Numbers shown are
    always the real computed ones either way; only the prose generation
    differs.
"""
from __future__ import annotations

import pandas as pd

from llm.provider import LLMProvider, SnowflakeCortexProvider


def _df_stats(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes("number").columns.tolist()
    stats = {"n_rows": len(df), "columns": list(df.columns)}
    if numeric_cols:
        metric_col = numeric_cols[-1]  # convention: metric is the last numeric column
        stats["metric_col"] = metric_col
        stats["total"] = df[metric_col].sum()
        stats["max_row"] = df.iloc[df[metric_col].idxmax()].to_dict() if len(df) else {}
        stats["min_row"] = df.iloc[df[metric_col].idxmin()].to_dict() if len(df) else {}
    return stats


def summarize_result(df: pd.DataFrame, question: str, provider: LLMProvider) -> str:
    """Summarize a single NL-query result table in 2-4 sentences."""
    if df is None or df.empty:
        return "The query returned no rows."

    stats = _df_stats(df)

    if isinstance(provider, SnowflakeCortexProvider) and provider.available():
        table_preview = df.head(20).to_csv(index=False)
        prompt = (
            f"A user asked: \"{question}\"\n\n"
            f"Here is the resulting data (CSV, up to 20 rows):\n{table_preview}\n\n"
            "Write a concise 2-4 sentence plain-language summary of the key finding "
            "for a healthcare claims analyst. Do not invent numbers not in the data."
        )
        return provider.complete(prompt)

    # Local templated fallback -- built directly from computed stats.
    label_col = next((c for c in df.columns if c != stats.get("metric_col")), None)
    lines = [f"This result has {stats['n_rows']} row(s)."]
    if "metric_col" in stats and label_col:
        top = stats["max_row"]
        lines.append(
            f"The highest value is {label_col.replace('_', ' ').lower()} "
            f"'{top.get(label_col)}' at {top.get(stats['metric_col']):,.2f} "
            f"({stats['metric_col'].replace('_', ' ').lower()})."
        )
        lines.append(f"Total across all rows: {stats['total']:,.2f}.")
    return " ".join(lines)


def executive_report(claims_summary: dict, fraud_summary: dict, provider: LLMProvider) -> str:
    """
    High-level narrative report combining portfolio stats and fraud-detection
    output, for the Streamlit "Reports" tab.

    claims_summary: dict with total_claims, total_billed, total_paid,
                     top_payer, denial_rate
    fraud_summary:  dict with n_high_risk, n_medium_risk, pct_high_risk,
                     top_fraud_reason_counts (dict[str,int])
    """
    if isinstance(provider, SnowflakeCortexProvider) and provider.available():
        prompt = (
            f"Claims portfolio stats: {claims_summary}\n"
            f"Fraud/anomaly detection stats: {fraud_summary}\n\n"
            "Write a short executive summary (4-6 sentences) for a healthcare "
            "analytics leader covering overall claims volume/spend and the "
            "fraud risk findings. Be specific with the numbers given; do not "
            "invent any not provided."
        )
        return provider.complete(prompt)

    # Local templated fallback.
    parts = [
        f"Over the period analyzed, {claims_summary['total_claims']:,} claims were filed "
        f"totaling ${claims_summary['total_billed']:,.0f} billed and "
        f"${claims_summary['total_paid']:,.0f} paid.",
        f"{claims_summary['top_payer']} is the largest payer by billed amount, and the "
        f"overall denial rate is {claims_summary['denial_rate']:.1%}.",
        f"The anomaly detection model flagged {fraud_summary['n_high_risk']:,} claims as "
        f"high risk ({fraud_summary['pct_high_risk']:.1%} of all claims) and "
        f"{fraud_summary['n_medium_risk']:,} as medium risk.",
    ]
    if fraud_summary.get("top_fraud_reason_counts"):
        top_reason = max(fraud_summary["top_fraud_reason_counts"].items(), key=lambda kv: kv[1])
        parts.append(
            f"The most common driver of high-risk flags was '{top_reason[0]}' "
            f"({top_reason[1]:,} claims)."
        )
    parts.append(
        "Recommend prioritizing high-risk claims for manual review, starting with the "
        "providers contributing the most flagged claims."
    )
    return " ".join(parts)

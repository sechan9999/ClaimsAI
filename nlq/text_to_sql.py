"""
Natural-language-to-SQL, grounded in the semantic model.

This is the LOCAL fallback implementation. In Snowflake, this entire module
is replaced by one REST call to the Cortex Analyst API
(POST /api/v2/cortex/analyst/message), which takes the same
semantic-model YAML (uploaded to a stage) plus the user's question and
returns SQL directly, using a full LLM rather than the keyword/verified-query
matching below. See README.md "Deploying to Snowflake" for the swap.

Approach used here, in order (cheapest/most reliable first):
  1. Verified-query match  -- reuse a hand-checked SQL template if the
     question closely resembles one in the semantic model.
  2. Slot-filling          -- detect a measure, an optional group-by
     dimension, and optional filters (payer/state/specialty/date range)
     directly from the semantic model's names/synonyms and the reference
     code lists, then assemble SQL.
  3. Give up honestly      -- tell the user which questions are supported
     locally rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from semantic.model import SemanticModel, Field

_JOIN_SQL = {
    "claims": "claims c",
    "providers": "JOIN providers p ON c.PROVIDER_ID = p.PROVIDER_ID",
    "patients": "JOIN patients pt ON c.PATIENT_ID = pt.PATIENT_ID",
}
_ALIAS = {"claims": "c", "providers": "p", "patients": "pt"}

_KNOWN_PAYERS = ["Medicare", "Medicaid", "Aetna", "UnitedHealthcare", "Cigna", "BCBS", "Humana"]
_KNOWN_STATUSES = ["Paid", "Denied", "Pending"]

_STOPWORDS = {
    "what", "is", "the", "are", "a", "an", "of", "for", "to", "in", "on", "by",
    "and", "or", "with", "was", "were", "how", "many", "much", "show", "me",
    "there", "did", "does", "do", "per", "total", "which", "than", "that",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


@dataclass
class NLQResult:
    sql: str
    explanation: str
    matched_via: str  # "verified_query" | "slot_filling" | "unresolved"
    params: list = None  # values bound to `?` placeholders in `sql`, if any

    def __post_init__(self):
        if self.params is None:
            self.params = []


def _best_verified_match(question: str, model: SemanticModel, threshold: float = 0.5):
    """
    Match on significant-word Jaccard overlap rather than raw character
    similarity -- char-level ratio (via difflib) gives misleadingly high
    scores to generic questions ("What is the ...") that share boilerplate
    phrasing but ask about something entirely different.
    """
    q_words = _significant_words(question)
    if not q_words:
        return None, 0.0
    best, best_score = None, 0.0
    for vq in model.verified_queries:
        vq_words = _significant_words(vq.question)
        if not vq_words:
            continue
        jaccard = len(q_words & vq_words) / len(q_words | vq_words)
        if jaccard > best_score:
            best, best_score = vq, jaccard
    if best and best_score >= threshold:
        return best, best_score
    return None, best_score


def _find_measure(question: str, model: SemanticModel):
    q = question.lower()
    # Pick the LONGEST matching name/synonym across all measures, not the
    # first one found -- e.g. "average billed amount" should prefer the
    # more specific "average billed amount" synonym on avg_billed_amount
    # over the shorter "billed" synonym on billed_amount.
    best_table, best_measure, best_len = None, None, 0
    for table_name, m in model.all_measures():
        candidates = [m.name.lower().replace("_", " ")] + [s.lower() for s in m.synonyms]
        for c in candidates:
            if c in q and len(c) > best_len:
                best_table, best_measure, best_len = table_name, m, len(c)
    if best_measure is not None:
        return best_table, best_measure
    # default measure if the question is clearly about money/volume but
    # didn't match a synonym exactly
    if any(w in q for w in ["cost", "amount", "spend", "money", "$"]):
        for table_name, m in model.all_measures():
            if m.name == "billed_amount":
                return table_name, m
    if any(w in q for w in ["how many", "number of", "count of", "claims"]):
        for table_name, m in model.all_measures():
            if m.name == "claim_count":
                return table_name, m
    return None, None


def _find_group_by(question: str, model: SemanticModel):
    q = question.lower()
    # Longest matching name/synonym wins, same rule as _find_measure -- e.g.
    # "providers" and "patients" both have a STATE dimension disambiguated
    # via the synonyms "provider state" / "patient state" in the semantic
    # model. Taking the first match (as this used to) meant a question like
    # "billed amount by provider state" could silently resolve to whichever
    # table's STATE field happened to iterate first, ignoring the qualifier.
    # Longest-match lets "provider state" beat the bare "state" on the other
    # table. A genuinely unqualified "by state" is still an honest tie,
    # broken by iteration order -- that ambiguity is inherent to the input,
    # not a matching bug.
    best_table, best_dim, best_len = None, None, 0
    for table_name, d in model.all_dimensions():
        candidates = [d.name.lower().replace("_", " ")] + [s.lower() for s in d.synonyms]
        for c in candidates:
            if c in q and len(c) > best_len:
                best_table, best_dim, best_len = table_name, d, len(c)
    return best_table, best_dim


def _find_filters(question: str) -> tuple[list[str], list]:
    """
    Returns (sql_fragments, params). Each fragment uses a `?` placeholder
    bound to the matching value in `params`, rather than the value being
    formatted into the fragment string. The values matched here only ever
    come from a fixed vocabulary (_KNOWN_PAYERS/_KNOWN_STATUSES) or a
    regex-extracted year, but binding them as params anyway means extending
    that vocabulary later can never reopen a SQL-injection surface.
    """
    fragments: list[str] = []
    params: list = []
    q_lower = question.lower()
    for payer in _KNOWN_PAYERS:
        if payer.lower() in q_lower:
            fragments.append("c.PAYER = ?")
            params.append(payer)
    for status in _KNOWN_STATUSES:
        if re.search(rf"\b{status.lower()}\b", q_lower):
            fragments.append("c.CLAIM_STATUS = ?")
            params.append(status)
    year_match = re.search(r"\b(20\d{2})\b", question)
    if year_match:
        fragments.append("EXTRACT(year FROM c.SERVICE_DATE) = ?")
        params.append(int(year_match.group(1)))
    return fragments, params


def _needs_table(table_name: str, group_table: str | None) -> bool:
    return table_name in ("providers", "patients") or group_table in ("providers", "patients")


def to_sql(question: str, model: SemanticModel) -> NLQResult:
    filters, filter_params = _find_filters(question)

    # 1. Verified-query shortcut -- only when the question carries no filter
    # cues (a specific payer, year, or status). Verified-query SQL is a fixed
    # template with no WHERE clause, so a filtered question always needs
    # slot-filling instead, even if the wording closely matches a template.
    if not filters:
        vq, score = _best_verified_match(question, model)
        if vq:
            return NLQResult(
                sql=vq.sql.strip(),
                explanation=f"Matched a verified query (\"{vq.question}\", similarity {score:.2f}).",
                matched_via="verified_query",
            )

    # 2. Slot filling
    measure_table, measure = _find_measure(question, model)
    if measure is None:
        return NLQResult(
            sql="",
            explanation=(
                "Couldn't confidently map this question to the semantic model in local mode. "
                "Try including a metric (e.g. 'billed amount', 'claim count', 'paid amount') "
                "and, optionally, a dimension to group by (e.g. 'by payer', 'by provider', "
                "'by month'). In production, Cortex Analyst handles open-ended phrasing here."
            ),
            matched_via="unresolved",
        )

    group_table, group_dim = _find_group_by(question, model)

    agg = (measure.default_aggregation or "sum").upper()
    agg = {"AVG": "AVG", "SUM": "SUM", "COUNT": "COUNT"}.get(agg, "SUM")
    measure_expr = f"c.{measure.expr}" if measure_table == "claims" else f"{_ALIAS[measure_table]}.{measure.expr}"

    select_parts = []
    group_parts = []
    joins = ["FROM claims c"]

    if group_dim is not None:
        alias = _ALIAS.get(group_table, "c")
        col = f"{alias}.{group_dim.expr}"
        select_parts.append(f"{col} AS {group_dim.name.upper()}")
        group_parts.append(col)
        if group_table == "providers":
            joins.append(_JOIN_SQL["providers"])
        elif group_table == "patients":
            joins.append(_JOIN_SQL["patients"])

    metric_alias = measure.name.upper()
    select_parts.append(f"{agg}({measure_expr}) AS {metric_alias}")

    if measure_table == "providers" and "providers" not in " ".join(joins):
        joins.append(_JOIN_SQL["providers"])
    if measure_table == "patients" and "patients" not in " ".join(joins):
        joins.append(_JOIN_SQL["patients"])

    where_clause = f"\nWHERE {' AND '.join(filters)}" if filters else ""
    group_clause = f"\nGROUP BY {', '.join(str(i + 1) for i in range(len(group_parts)))}" if group_parts else ""
    order_clause = f"\nORDER BY {metric_alias} DESC" if group_parts else ""
    limit_clause = "\nLIMIT 25" if group_parts else ""

    sql = (
        f"SELECT {', '.join(select_parts)}\n"
        + "\n".join(joins)
        + where_clause
        + group_clause
        + order_clause
        + limit_clause
        + ";"
    )

    explanation_bits = [f"metric = {measure.name} ({agg})"]
    if group_dim is not None:
        explanation_bits.append(f"grouped by {group_dim.name}")
    if filters:
        explanation_bits.append(f"filtered on {', '.join(filters)}")
    explanation = "Built from the semantic model: " + "; ".join(explanation_bits) + "."

    return NLQResult(sql=sql, explanation=explanation, matched_via="slot_filling", params=filter_params)

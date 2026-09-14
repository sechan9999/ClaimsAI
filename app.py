"""
ClaimsAI -- Streamlit app for patient-claims analytics with a GenAI layer.

Run locally:   streamlit run app.py
Deploy target: Streamlit in Snowflake (see README.md "Deploying to Snowflake")

Tabs:
  - Ask            natural-language query over claims, grounded in the
                    semantic model, with an auto-generated narrative summary.
  - Fraud Detection ranked, explained anomaly scores per claim.
  - Reports         executive-level portfolio summary and trend charts.
  - Data Explorer   raw table browsing with filters.
  - Semantic Model  what's defined in the semantic layer (for transparency /
                    handoff to whoever registers this with Cortex Analyst).
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from semantic.model import load_semantic_model
from nlq.text_to_sql import to_sql
from warehouse.connection import run_query
from fraud.detection import score_claims, claim_detail
from fraud.benford import provider_benford_scores
from reporting.summarize import summarize_result, executive_report
from llm.provider import get_provider

st.set_page_config(page_title="ClaimsAI", page_icon="🩺", layout="wide")


@st.cache_resource
def _model():
    return load_semantic_model()


@st.cache_resource
def _provider():
    # Pass a live Snowpark session here once running inside Streamlit-in-Snowflake:
    #   from snowflake.snowpark.context import get_active_session
    #   return get_provider(get_active_session())
    return get_provider()


@st.cache_data
def _run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    # params is a tuple (not a list) so st.cache_data can hash it as part of
    # the cache key -- see nlq/text_to_sql.py / warehouse/connection.py for
    # why values are bound here rather than formatted into `sql`.
    return run_query(sql, params=params)


@st.cache_data
def _scored_claims() -> pd.DataFrame:
    return score_claims()


@st.cache_data
def _benford_scores() -> pd.DataFrame:
    return provider_benford_scores()


semantic_model = _model()
provider = _provider()

st.title("🩺 ClaimsAI")
st.caption(
    "Patient claims analytics with a Snowflake-portable semantic model and GenAI layer. "
    "Running on synthetic data in local demo mode."
    if not getattr(provider, "session", None)
    else "Patient claims analytics — connected to Snowflake Cortex."
)

tab_ask, tab_fraud, tab_reports, tab_explore, tab_semantic = st.tabs(
    ["💬 Ask", "🚩 Fraud Detection", "📊 Reports", "🔎 Data Explorer", "🧭 Semantic Model"]
)

# ----------------------------------------------------------------- Ask tab
with tab_ask:
    st.subheader("Ask a question about the claims data")
    example_qs = [q.question for q in semantic_model.verified_queries]
    st.caption("Try: " + " · ".join(f"“{q}”" for q in example_qs))

    question = st.text_input("Question", placeholder="e.g. What is the total billed amount by payer?")
    if question:
        result = to_sql(question, semantic_model)
        if result.matched_via == "unresolved":
            st.warning(result.explanation)
        else:
            with st.expander("Generated SQL", expanded=False):
                st.code(result.sql, language="sql")
                st.caption(result.explanation)
            try:
                df = _run_query(result.sql, tuple(result.params))
            except Exception as e:
                st.error(f"Query failed: {e}")
                df = pd.DataFrame()

            if not df.empty:
                col_table, col_chart = st.columns([1, 1])
                with col_table:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                with col_chart:
                    numeric_cols = df.select_dtypes("number").columns.tolist()
                    label_cols = [c for c in df.columns if c not in numeric_cols]
                    if numeric_cols and label_cols and len(df) > 1:
                        fig = px.bar(df, x=label_cols[0], y=numeric_cols[-1])
                        st.plotly_chart(fig, use_container_width=True)
                    elif numeric_cols and len(label_cols) == 0 and "MONTH" in df.columns:
                        fig = px.line(df, x=df.columns[0], y=numeric_cols[-1])
                        st.plotly_chart(fig, use_container_width=True)

                st.info(summarize_result(df, question, provider))
            else:
                st.write("No rows returned.")

# ------------------------------------------------------------- Fraud tab
with tab_fraud:
    st.subheader("Anomaly-ranked claims")
    st.caption(
        "Unsupervised IsolationForest over peer-group billing-amount deviation, "
        "same-day claim duplication, 30-day claim frequency, and submission lag — "
        "computed entirely in-warehouse (sql/002_claim_features.sql)."
    )

    scored = _scored_claims()
    tier_filter = st.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    filtered = scored[scored["RISK_TIER"].isin(tier_filter)]

    c1, c2, c3 = st.columns(3)
    c1.metric("High risk claims", int((scored["RISK_TIER"] == "High").sum()))
    c2.metric("Medium risk claims", int((scored["RISK_TIER"] == "Medium").sum()))
    c3.metric("Total claims scored", len(scored))

    st.dataframe(
        filtered[
            ["CLAIM_ID", "PATIENT_ID", "PROVIDER_ID", "PROCEDURE_CODE", "BILLED_AMOUNT",
             "RISK_SCORE", "RISK_TIER", "EXPLANATION"]
        ].head(300),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Provider billing digit-pattern (Benford's Law)")
    st.caption(
        "Group-level signal, independent of the per-claim model above: tests whether each "
        "provider's billed-amount leading digits follow Benford's Law (chi-square goodness-of-fit). "
        "A significant deviation means the provider's amounts, as a group, look statistically "
        "manufactured rather than naturally varied — worth a review queue of its own, even when "
        "no single claim from that provider ranks as a per-claim outlier above."
    )
    benford = _benford_scores()
    n_flagged = int(benford["BENFORD_FLAG"].sum()) if not benford.empty else 0
    st.metric("Providers flagged (p < 0.01)", n_flagged, help=f"Out of {len(benford)} providers with enough claims to test")
    if n_flagged:
        st.dataframe(
            benford[benford["BENFORD_FLAG"]][["PROVIDER_ID", "N_CLAIMS", "CHI2_STATISTIC", "P_VALUE"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Investigate a claim")
    pick = st.text_input("Claim ID", placeholder="e.g. CL0000123")
    if pick:
        try:
            detail = claim_detail(pick.strip())
            st.json(detail["claim"])
            st.write("**Risk score:**", detail["features"]["RISK_SCORE"],
                      "| **Tier:**", detail["features"]["RISK_TIER"])
            st.write("**Why flagged:**", detail["features"]["EXPLANATION"])
        except KeyError as e:
            st.error(str(e))

# ----------------------------------------------------------- Reports tab
with tab_reports:
    st.subheader("Executive summary")

    totals = _run_query(
        "SELECT COUNT(*) AS N, SUM(BILLED_AMOUNT) AS BILLED, SUM(PAID_AMOUNT) AS PAID FROM claims"
    ).iloc[0]
    top_payer = _run_query(
        "SELECT PAYER FROM claims GROUP BY PAYER ORDER BY SUM(BILLED_AMOUNT) DESC LIMIT 1"
    ).iloc[0]["PAYER"]
    denial_rate = _run_query(
        "SELECT AVG(CASE WHEN CLAIM_STATUS='Denied' THEN 1.0 ELSE 0 END) AS X FROM claims"
    ).iloc[0]["X"]

    scored = _scored_claims()
    claims_summary = {
        "total_claims": int(totals["N"]),
        "total_billed": float(totals["BILLED"]),
        "total_paid": float(totals["PAID"]),
        "top_payer": top_payer,
        "denial_rate": float(denial_rate),
    }
    fraud_summary = {
        "n_high_risk": int((scored["RISK_TIER"] == "High").sum()),
        "n_medium_risk": int((scored["RISK_TIER"] == "Medium").sum()),
        "pct_high_risk": float((scored["RISK_TIER"] == "High").mean()),
        "top_fraud_reason_counts": {},
    }

    st.info(executive_report(claims_summary, fraud_summary, provider))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total claims", f"{claims_summary['total_claims']:,}")
    c2.metric("Total billed", f"${claims_summary['total_billed']:,.0f}")
    c3.metric("Total paid", f"${claims_summary['total_paid']:,.0f}")
    c4.metric("Denial rate", f"{claims_summary['denial_rate']:.1%}")

    monthly = _run_query(
        "SELECT DATE_TRUNC('month', SERVICE_DATE) AS MONTH, SUM(BILLED_AMOUNT) AS BILLED "
        "FROM claims GROUP BY 1 ORDER BY 1"
    )
    st.plotly_chart(px.line(monthly, x="MONTH", y="BILLED", title="Billed amount by month"),
                     use_container_width=True)

    by_payer = _run_query(
        "SELECT PAYER, SUM(BILLED_AMOUNT) AS BILLED FROM claims GROUP BY PAYER ORDER BY BILLED DESC"
    )
    st.plotly_chart(px.bar(by_payer, x="PAYER", y="BILLED", title="Billed amount by payer"),
                     use_container_width=True)

# ------------------------------------------------------- Data explorer tab
with tab_explore:
    st.subheader("Browse raw tables")
    table = st.selectbox("Table", ["claims", "providers", "patients"])
    limit = st.slider("Rows to show", 10, 1000, 100)
    st.dataframe(_run_query(f"SELECT * FROM {table} LIMIT {limit}"), use_container_width=True, hide_index=True)

# ------------------------------------------------------- Semantic model tab
with tab_semantic:
    st.subheader("Semantic model")
    st.caption(
        "Defined once in semantic/claims_semantic_model.yaml (Cortex-Analyst-compatible format). "
        "The Ask tab's SQL generation and this view both read the same file."
    )
    for tname, t in semantic_model.tables.items():
        with st.expander(f"**{tname}** — {t.description}", expanded=False):
            if t.dimensions or t.time_dimensions:
                st.markdown("**Dimensions**")
                st.table(pd.DataFrame(
                    [{"name": d.name, "expr": d.expr, "synonyms": ", ".join(d.synonyms)}
                     for d in t.all_dimensions()]
                ))
            if t.measures:
                st.markdown("**Measures**")
                st.table(pd.DataFrame(
                    [{"name": m.name, "aggregation": m.default_aggregation, "synonyms": ", ".join(m.synonyms)}
                     for m in t.measures]
                ))

    st.markdown("**Verified queries**")
    st.table(pd.DataFrame([{"question": v.question} for v in semantic_model.verified_queries]))

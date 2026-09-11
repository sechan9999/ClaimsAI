# ClaimsAI

A GenAI-powered patient-claims analytics prototype, built to run locally on
synthetic data and deploy as **Streamlit in Snowflake** with **Cortex
Analyst** / **Cortex Complete** with minimal changes.

Inspired by the pattern in Snowflake apps like the linked CreditAI
Streamlit-in-Snowflake example: a semantic layer + natural-language query
interface over a domain dataset. This applies the same pattern to patient
claims, and adds fraud/anomaly detection and executive reporting.

## What it does

- **Ask** — natural-language questions over claims data ("What is the total
  billed amount by payer?"), answered via a semantic model rather than
  free-form text-to-SQL, so results stay auditable and consistent.
- **Fraud Detection** — every claim is scored for anomalous billing patterns
  (unsupervised IsolationForest over peer-group amount deviation, duplicate
  same-day billing, excess visit frequency, and submission-lag outliers),
  each with a plain-language explanation of why it was flagged.
- **Reports** — an auto-generated executive summary plus trend charts.
- **Data Explorer** — raw table browsing.
- **Semantic Model** — a transparent view of the tables/measures/dimensions
  driving the Ask tab, matching what would be registered with Cortex Analyst.

## Architecture

```
data/        synthetic patient claims generator (patients, providers, claims)
semantic/    Cortex-Analyst-format semantic model (YAML) + Python loader
sql/         ANSI SQL transforms: curated aggregates, per-claim risk features
warehouse/   the ONE module that talks to the data warehouse (DuckDB locally)
nlq/         natural-language -> SQL, grounded in the semantic model
fraud/       anomaly scoring + explanation generation
reporting/   narrative summarization (LLM-backed or templated fallback)
llm/         pluggable LLM provider (Snowflake Cortex, or local fallback)
app.py       Streamlit UI tying it all together
```

The design principle: **one semantic model, one warehouse abstraction.**
Everything else (NL query, fraud detection, reporting) reads through those
two seams, so moving from local DuckDB to real Snowflake — or from templated
text to live Cortex-generated narrative — touches only `warehouse/connection.py`
and `llm/provider.py`, never the app logic itself.

## Running locally

```bash
pip install -r requirements.txt
python data/generate_synthetic_claims.py   # writes data/generated/*.parquet
streamlit run app.py
```

Optional: validate the fraud model against the synthetic ground-truth labels
(not part of the app — dev-only sanity check):

```bash
python -m fraud.validate_against_labels
```
On the generated sample: ~99% precision / ~92% recall in the top 3,000
risk-ranked claims. The detector is strongest on outlier-amount, duplicate,
and excess-frequency patterns; unbundling (a subtler ~2x amount inflation)
is caught less often (~28%) — worth strengthening with a provider-level
billing-mix feature before relying on this for real triage.

## Deploying to Snowflake

Three swap points, each already isolated to one file:

**1. Data → real Snowflake tables.**
Load `patients`, `providers`, `claims` into `HEALTHCARE_DB.CLAIMS` (the
database/schema names already used in `semantic/claims_semantic_model.yaml`).
Column names are written in `SCREAMING_SNAKE_CASE` throughout specifically so
they land in Snowflake unchanged.

**2. Warehouse connection → Snowpark session.**
Replace `warehouse/connection.py`'s `run_query()` with the Snowpark version
sketched in that file's docstring, or — simplest — when running as
Streamlit-in-Snowflake, use `get_active_session()` and call `.sql(sql).to_pandas()`.
Nothing outside this file needs to change.

**3. NL query → Cortex Analyst API.**
Upload `semantic/claims_semantic_model.yaml` to a Snowflake stage and register
it with Cortex Analyst
([docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)).
Replace the call in `app.py`'s Ask tab (`to_sql(question, semantic_model)`)
with a POST to `/api/v2/cortex/analyst/message`, passing the same question and
semantic model stage path — Cortex Analyst returns SQL directly, with far
better open-ended language coverage than the local keyword/verified-query
matcher in `nlq/text_to_sql.py`. The YAML file needs no changes to make this
switch.

**4. Narrative text → Cortex Complete.**
`llm/provider.py` already includes `SnowflakeCortexProvider`, which calls
`SNOWFLAKE.CORTEX.COMPLETE()` over a live Snowpark session. In `app.py`,
change:
```python
return get_provider()
```
to:
```python
from snowflake.snowpark.context import get_active_session
return get_provider(get_active_session())
```
`reporting/summarize.py` already branches on provider type, so real narrative
generation turns on with no other changes.

## Known limitations (by design, for a prototype)

- The synthetic data is illustrative (a small ICD-10/CPT slice), not a real
  code set or real PHI.
- Local NL-to-SQL is keyword/synonym matching against the semantic model —
  intentionally simple, since Cortex Analyst is the intended production path.
- The fraud model is unsupervised and untuned beyond what validation showed;
  treat its output as a triage prioritization aid, not an adjudication.

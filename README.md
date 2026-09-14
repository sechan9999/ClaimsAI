# 🩺 ClaimsAI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Snowflake Cortex](https://img.shields.io/badge/Snowflake-Cortex%20Ready-29B5E8?logo=snowflake&logoColor=white)](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.0+-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

**ClaimsAI** is an enterprise-grade patient-claims analytics and fraud detection prototype featuring a Snowflake-portable semantic layer and a GenAI intelligence copilot. Built to develop and run locally on synthetic data with DuckDB, it seamlessly deploys to **Streamlit in Snowflake (SiS)** with **Cortex Analyst** and **Cortex Complete** with zero application logic rewrites.

---

## 📑 Table of Contents

- [Core Capabilities](#-core-capabilities)
- [Architecture & Design Principles](#-architecture--design-principles)
- [Repository Structure](#-repository-structure)
- [Getting Started Locally](#-getting-started-locally)
- [Module Deep Dives](#-module-deep-dives)
  - [1. Semantic Layer & Cortex-Compatible Model](#1-semantic-layer--cortex-compatible-model)
  - [2. Natural Language Query (NLQ) & Parameterized SQL](#2-natural-language-query-nlq--parameterized-sql)
  - [3. In-Warehouse Feature Engineering & Fraud Scoring](#3-in-warehouse-feature-engineering--fraud-scoring)
  - [4. Narrative Reporting & Pluggable LLMs](#4-narrative-reporting--pluggable-llms)
- [Deploying to Snowflake](#-deploying-to-snowflake)
- [Validation & Benchmarks](#-validation--benchmarks)
- [Design Considerations & Roadmap](#-design-considerations--roadmap)

---

## 🌟 Core Capabilities

| Feature | Description |
| :--- | :--- |
| 💬 **Ask (Semantic NLQ)** | Ask plain-English questions (*"What is the total billed amount by payer?"* or *"Show me denied claims in 2024"*). Grounded directly in the Cortex Analyst semantic model using longest-match disambiguation and safe parameterized execution. Automatically renders interactive Plotly charts and narrative insights. |
| 🚩 **Fraud & Abuse Detection** | Unsupervised `IsolationForest` scoring combined with in-warehouse feature engineering (`sql/002_claim_features.sql`). Analyzes peer-group z-score anomalies, duplicate same-day filings, 30-day visit velocity, and submission lag. Assigns a calibrated 0–100 risk score, risk tier (`High`, `Medium`, `Low`), and plain-English root-cause explanations. |
| 📊 **Executive Reporting** | Portfolio-level financial health dashboard tracking Claim Volume, Billed Spend, Reimbursed Amount, Denial Rates, and Fraud Exposure. Features monthly trend trajectory graphs and automated narrative briefings. |
| 🔎 **Data Explorer** | Direct warehouse table browser with row sampling and column sorting across raw `claims`, `patients`, and `providers` tables. |
| 🧭 **Semantic Model Inspector** | Complete transparency view into entities, dimensions, calculated measures, synonyms, and verified golden queries configured in the semantic layer. |

---

## 🏛 Architecture & Design Principles

```
                              ┌──────────────────────────────────────────────┐
                              │            Streamlit User Interface          │
                              │   Ask  |  Fraud  |  Reports  |  Explorer     │
                              └──────┬──────────────────┬──────────────┬─────┘
                                     │                  │              │
                   ┌─────────────────┴──┐       ┌───────┴──────┐  ┌────┴────────────────┐
                   │    NLQ Engine      │       │ Fraud Engine │  │ Reporting Engine    │
                   │  (text_to_sql.py)  │       │(detection.py)│  │   (summarize.py)    │
                   └─────────┬──────────┘       └───────┬──────┘  └─────┬───────────────┘
                             │                          │               │
                             ▼                          ▼               │
               ┌───────────────────────────┐  ┌──────────────────┐      │
               │    Semantic Model Layer   │  │ Feature Pipeline │      │
               │(claims_semantic_model.yml)│  │(002_features.sql)│      │
               └─────────────┬─────────────┘  └─────────┬────────┘      │
                             │                          │               │
                             ▼                          ▼               ▼
               ┌─────────────────────────────────────────────────┐ ┌────────────────────┐
               │         Unified Warehouse Abstraction           │ │ Pluggable LLM      │
               │             warehouse/connection.py             │ │ Cortex / Local Fall│
               └─────────────┬─────────────────────┬─────────────┘ └────────────────────┘
                             │                     │
                    Local Dev Engine        Production Engine
                     [ DuckDB 1.0+ ]     [ Snowflake / Snowpark ]
```

### Core Design Principles:
1. **One Semantic Model, One Warehouse Abstraction**: All application logic communicates through `semantic/model.py` and `warehouse/connection.py`. Switching from DuckDB to Snowflake requires no UI or analytical code changes.
2. **Strict SQL Parameterization**: Dynamic filters (payer, status, year, claim ID) are bound through native driver parameters rather than string concatenation, protecting against injection and enabling deterministic query caching in Streamlit (`st.cache_data`).
3. **Auditable & Explainable AI**: Text-to-SQL is strictly grounded in authorized dimensions and verified queries, eliminating hallucinations. Fraud predictions pair mathematical anomaly metrics with clear, natural language triggers.

---

## 📂 Repository Structure

```text
claimsai/
├── app.py                          # Main Streamlit application (5 interactive tabs)
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── data/
│   └── generate_synthetic_claims.py # Synthetic claims generator with controlled fraud
├── semantic/
│   ├── claims_semantic_model.yaml  # Cortex Analyst-compatible YAML specification
│   └── model.py                    # Semantic model parser and dataclasses
├── sql/
│   ├── 001_curated_provider_monthly.sql # Monthly provider aggregate transform
│   └── 002_claim_features.sql      # In-warehouse peer-group and behavioral features
├── warehouse/
│   └── connection.py               # Database abstraction layer (DuckDB local / Snowpark target)
├── nlq/
│   └── text_to_sql.py              # Semantic-grounded NLQ with longest-match & param binding
├── fraud/
│   ├── detection.py                # IsolationForest scoring, risk calibration & explanations
│   └── validate_against_labels.py  # Precision/recall benchmarking script
├── reporting/
│   └── summarize.py                # Narrative summarization (Cortex Complete or heuristic fallback)
└── llm/
    └── provider.py                 # Pluggable LLM interface (SnowflakeCortexProvider / LocalFallback)
```

---

## 🚀 Getting Started Locally

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10+ installed:

```bash
# Clone the repository
git clone https://github.com/sechan9999/ClaimsAI.git
cd ClaimsAI

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Claims Data

Generate realistic healthcare claims, patient cohorts, and provider records with embedded billing anomalies:

```bash
python data/generate_synthetic_claims.py
```
*Output*: Generates Parquet files under `data/generated/` for `patients.parquet`, `providers.parquet`, `claims.parquet`, and evaluation labels `fraud_labels.parquet`.

### 3. Launch the Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔍 Module Deep Dives

### 1. Semantic Layer & Cortex-Compatible Model

The semantic model in `semantic/claims_semantic_model.yaml` implements the official [Snowflake Cortex Analyst specification](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst):
- **Base Tables**: Configured with uppercase schema identifiers (`HEALTHCARE_DB.CLAIMS.CLAIMS`, `PROVIDERS`, `PATIENTS`).
- **Measures & Aggregations**: Predefined calculation definitions (`claim_count`, `billed_amount`, `paid_amount`, `avg_billed_amount`, `denial_rate`).
- **Dimensions & Synonyms**: Rich entity mapping with comprehensive synonyms (e.g. `payer` -> *insurer*, *health plan*; `procedure_code` -> *CPT*, *procedure*).
- **Verified Queries**: Pre-tested golden questions mapped directly to approved SQL statements.

### 2. Natural Language Query (NLQ) & Parameterized SQL

`nlq/text_to_sql.py` resolves natural language into executable SQL using the semantic model:
- **Verified Query Matching**: Evaluates exact and fuzzy question matches against verified queries first.
- **Longest-Match Disambiguation**: Uses longest-match substring matching across dimension synonyms so qualified terms like `"provider state"` and `"patient state"` correctly route to distinct tables without collision.
- **Secure Parameter Binding**: Extracted filter values (e.g. `payer = ?`, `year = ?`, `status = ?`) are populated into parameterized queries:
  ```python
  # Returned NLQResult
  sql = "SELECT PAYER, SUM(BILLED_AMOUNT) AS BILLED FROM claims WHERE PAYER = ? GROUP BY PAYER"
  params = ("Medicare",)
  ```

### 3. In-Warehouse Feature Engineering & Fraud Scoring

`sql/002_claim_features.sql` computes behavioral risk features directly in SQL:
1. **`AMOUNT_Z_IN_PEER_GROUP`**: Standardized deviation of billed charges against peer claims sharing the same procedure code:
   $$\frac{\text{BILLED\_AMOUNT} - \mu_{\text{CPT}}}{\sigma_{\text{CPT}}}$$
2. **`PATIENT_PROVIDER_DAILY_COUNT`**: Detects duplicate or split billing submitted for the same patient on the same day.
3. **`PATIENT_PROVIDER_30D_COUNT`**: Rolling 30-day window visit frequency flagging excessive encounter velocity.
4. **`SUBMISSION_LAG_DAYS`**: Elapsed days between date of service and claim submission date.

In `fraud/detection.py`, an unsupervised **IsolationForest** consumes these vectors and converts decision boundaries into a calibrated 0–100 **Risk Score**, categorizing claims into `High`, `Medium`, or `Low` tiers alongside plain-language triage explanations.

### 4. Narrative Reporting & Pluggable LLMs

`reporting/summarize.py` and `llm/provider.py` deliver fluent analytical commentary:
- **Local Mode**: Uses robust template heuristics to generate clear, immediate executive summaries of query results and portfolio trends.
- **Snowflake Cortex Mode**: Uses `SNOWFLAKE.CORTEX.COMPLETE()` (e.g. with `mistral-large` or `llama3-70b`) to produce nuanced natural-language briefings.

---

## ❄️ Deploying to Snowflake

ClaimsAI is structured with clean separation of concerns for frictionless deployment to **Streamlit in Snowflake (SiS)**:

### 1. Data Ingestion
Upload `data/generated/*.parquet` into a Snowflake database table (`HEALTHCARE_DB.CLAIMS`):
```sql
CREATE DATABASE IF NOT EXISTS HEALTHCARE_DB;
CREATE SCHEMA IF NOT EXISTS HEALTHCARE_DB.CLAIMS;
-- Load parquet data into CLAIMS, PATIENTS, and PROVIDERS
```
*Note: Column names across all schemas are in `SCREAMING_SNAKE_CASE` to match Snowflake defaults without quoting.*

### 2. Activate Warehouse Connection
In `warehouse/connection.py`, toggle `run_query` to use the active Snowpark session:
```python
from snowflake.snowpark.context import get_active_session

def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    session = get_active_session()
    # Execute parameterized query via Snowpark
    return session.sql(sql, params=list(params)).to_pandas()
```

### 3. Register Semantic Model with Cortex Analyst
Upload `semantic/claims_semantic_model.yaml` to an internal Snowflake stage:
```sql
CREATE STAGE IF NOT EXISTS HEALTHCARE_DB.CLAIMS.SEMANTIC_MODELS;
PUT file://semantic/claims_semantic_model.yaml @HEALTHCARE_DB.CLAIMS.SEMANTIC_MODELS AUTO_COMPRESS=FALSE;
```
In `app.py`, route the Ask tab to Cortex Analyst via `/api/v2/cortex/analyst/message`.

### 4. Connect Cortex Complete
Update `_provider()` in `app.py`:
```python
from snowflake.snowpark.context import get_active_session
from llm.provider import SnowflakeCortexProvider

@st.cache_resource
def _provider():
    return SnowflakeCortexProvider(session=get_active_session(), model="mistral-large2")
```

---

## 📊 Validation & Benchmarks

To validate the unsupervised fraud detection engine against synthetic ground-truth anomalies:

```bash
python -m fraud.validate_against_labels
```

### Benchmark Results (Synthetic Evaluation Dataset):
- **Top 3,000 Scored Claims**: **~99% Precision** / **~92% Recall**
- **High-Detection Patterns**:
  - Outlier billed amounts (peer z-score $\ge 3$)
  - Duplicate same-day billing
  - Abnormal 30-day visit velocity
- **Subtler Patterns (e.g. ~2x Unbundling)**: Detected at ~28% recall; can be further strengthened with provider-level billing mixture features prior to production deployment.

---

## 🛡️ License

This project is licensed under the MIT License.

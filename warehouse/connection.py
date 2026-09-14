"""
Local warehouse connection.

This is the ONE module you swap to move the app from local DuckDB to real
Snowflake. Everywhere else in the app calls `run_query(sql)` -- it never
talks to DuckDB or Snowflake directly. DuckDB was chosen locally because it
executes the same ANSI SQL dialect used in sql/*.sql without a live warehouse.

--- To point this at Snowflake instead ---
Replace get_connection()/run_query() with:

    import snowflake.connector
    def get_connection():
        return snowflake.connector.connect(
            account=..., user=..., role=..., warehouse=..., database="HEALTHCARE_DB", schema="CLAIMS",
        )
    def run_query(sql, params=None) -> pd.DataFrame:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params or {})
            return cur.fetch_pandas_all()

Or, when running as Streamlit-in-Snowflake, use `st.connection("snowflake")`
/ `get_active_session()` (Snowpark) instead of a connector object -- the
`run_query` signature below stays identical either way.

--- Parameters ---
Never build SQL by f-string-ing user-controlled values into the query text.
`run_query(sql, params)` passes `params` straight to the underlying driver's
own parameterization, so untrusted input (a claim ID typed into a text box,
a payer name matched out of a free-text question, etc.) is always bound as
data, never spliced into the SQL string. DuckDB and snowflake-connector-python
both accept `?` positional placeholders, so callers write the same SQL either
way; only the driver underneath changes.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import duckdb
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

_TABLE_FILES = {
    "claims": "claims.parquet",
    "providers": "providers.parquet",
    "patients": "patients.parquet",
}

_conn = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = duckdb.connect(database=":memory:")
        for table_name, filename in _TABLE_FILES.items():
            path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run `python data/generate_synthetic_claims.py` first."
                )
            _conn.execute(
                f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet('{path}')"
            )
    return _conn


def run_query(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    """
    Execute SQL against the warehouse and return a pandas DataFrame.

    Pass untrusted values (claim IDs, matched filter values, etc.) via
    `params` with `?` placeholders in `sql` -- never by formatting them into
    the SQL string. e.g. run_query("SELECT * FROM claims WHERE CLAIM_ID = ?", [claim_id])
    """
    conn = get_connection()
    if params:
        return conn.execute(sql, list(params)).fetchdf()
    return conn.execute(sql).fetchdf()


def run_sql_file(path: str) -> pd.DataFrame:
    with open(path) as fh:
        sql = fh.read()
    return run_query(sql)


def table_columns(table_name: str) -> list[str]:
    df = run_query(f"SELECT * FROM {table_name} LIMIT 0")
    return list(df.columns)

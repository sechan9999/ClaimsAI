"""
Pluggable LLM provider layer.

Mirrors the graceful-degradation pattern from the RxHCC fraud detection app:
try a real LLM backend, fall back to a deterministic rule-based/templated
response if none is configured, so the app is always usable in a demo even
without live credentials.

Providers:
  - SnowflakeCortexProvider : calls SNOWFLAKE.CORTEX.COMPLETE() via a live
    Snowpark session. This is the production path once deployed as
    Streamlit-in-Snowflake -- no API key needed there, Cortex runs in-warehouse.
  - LocalTemplateProvider   : no model call at all. Produces clear, honest
    templated text from the actual computed statistics, so local demos never
    show fabricated numbers.

Swap-in point for a hosted LLM (Anthropic/OpenAI) during local development:
add a class implementing `.complete()` below and select it in get_provider().
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        ...

    @abstractmethod
    def available(self) -> bool:
        ...


class SnowflakeCortexProvider(LLMProvider):
    """
    Production provider. Requires a live Snowpark session (available for
    free inside Streamlit-in-Snowflake via `get_active_session()`).

    Usage once deployed in Snowflake:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        provider = SnowflakeCortexProvider(session, model="llama3.1-70b")
    """

    def __init__(self, session=None, model: str = "llama3.1-70b"):
        self.session = session
        self.model = model

    def available(self) -> bool:
        return self.session is not None

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        if not self.available():
            raise RuntimeError("No active Snowpark session configured for Cortex.")
        # SNOWFLAKE.CORTEX.COMPLETE(model, prompt) -> STRING
        escaped = prompt.replace("'", "''")
        sql = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{self.model}', '{escaped}') AS RESPONSE"
        row = self.session.sql(sql).collect()[0]
        return row["RESPONSE"]


class LocalTemplateProvider(LLMProvider):
    """
    Deterministic fallback used for local development / this prototype.
    Does not call any model -- composes plain-language text directly from
    the statistics it's given, via the small template functions in
    reporting/summarize.py and fraud/detection.py. Kept here as the
    "always available" provider so the rest of the app doesn't special-case
    the no-credentials path.
    """

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        # No generative call -- callers that need real templated narrative
        # should use the helpers in reporting/summarize.py instead of this
        # generic passthrough. This exists so `provider.complete(...)` never
        # raises, keeping the interface uniform.
        return (
            "[local demo mode: no live LLM connected] "
            "Connect a Snowflake session (SnowflakeCortexProvider) or another "
            "provider to generate free-text narrative here."
        )


def get_provider(snowpark_session=None) -> LLMProvider:
    """Single entry point the app uses to obtain a provider."""
    cortex = SnowflakeCortexProvider(session=snowpark_session)
    if cortex.available():
        return cortex
    return LocalTemplateProvider()

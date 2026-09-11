"""
Loader for the Cortex-Analyst-style semantic model YAML.

Parses semantic/claims_semantic_model.yaml into simple dataclasses that the
NL-to-SQL layer (nlq/text_to_sql.py) and the Streamlit "Semantic Model" tab
can both consume, without re-parsing YAML in multiple places.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
import yaml

MODEL_PATH = os.path.join(os.path.dirname(__file__), "claims_semantic_model.yaml")


@dataclass
class Field:
    name: str
    expr: str
    data_type: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    default_aggregation: str | None = None  # measures only


@dataclass
class Table:
    name: str
    description: str
    dimensions: list[Field]
    time_dimensions: list[Field]
    measures: list[Field]

    def all_dimensions(self) -> list[Field]:
        return self.dimensions + self.time_dimensions

    def find_field(self, token: str) -> Field | None:
        token = token.lower().strip()
        for f in self.all_dimensions() + self.measures:
            names = [f.name.lower()] + [s.lower() for s in f.synonyms]
            if token in names:
                return f
        return None


@dataclass
class VerifiedQuery:
    name: str
    question: str
    sql: str


@dataclass
class SemanticModel:
    name: str
    description: str
    tables: dict[str, Table]
    relationships: list[dict]
    verified_queries: list[VerifiedQuery]

    def all_measures(self) -> list[tuple[str, Field]]:
        return [(t.name, m) for t in self.tables.values() for m in t.measures]

    def all_dimensions(self) -> list[tuple[str, Field]]:
        return [(t.name, d) for t in self.tables.values() for d in t.all_dimensions()]


def _parse_fields(raw: list[dict] | None, is_measure: bool = False) -> list[Field]:
    out = []
    for r in raw or []:
        out.append(
            Field(
                name=r["name"],
                expr=r["expr"],
                data_type=r.get("data_type", "VARCHAR"),
                description=r.get("description", ""),
                synonyms=r.get("synonyms", []),
                default_aggregation=r.get("default_aggregation") if is_measure else None,
            )
        )
    return out


def load_semantic_model(path: str = MODEL_PATH) -> SemanticModel:
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    tables = {}
    for t in raw["tables"]:
        tables[t["name"]] = Table(
            name=t["name"],
            description=t.get("description", ""),
            dimensions=_parse_fields(t.get("dimensions")),
            time_dimensions=_parse_fields(t.get("time_dimensions")),
            measures=_parse_fields(t.get("measures"), is_measure=True),
        )

    verified = [
        VerifiedQuery(name=v["name"], question=v["question"], sql=v["sql"])
        for v in raw.get("verified_queries", [])
    ]

    return SemanticModel(
        name=raw["name"],
        description=raw.get("description", ""),
        tables=tables,
        relationships=raw.get("relationships", []),
        verified_queries=verified,
    )

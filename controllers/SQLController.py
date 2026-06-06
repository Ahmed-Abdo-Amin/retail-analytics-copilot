"""
SQLController: DSPy NL→SQL generation with schema-aware prompting.
"""
from __future__ import annotations
import json
from typing import Any, Dict

from agent.dspy_modules import NL2SQLModule
from controllers.BaseController import BaseController
from tools.schema_cache import cached_schema_string


class SQLController(BaseController):
    """Generates SQLite queries from natural language."""

    def __init__(self, nl2sql_module: NL2SQLModule):
        super().__init__("SQLController")
        self._module = nl2sql_module

    def generate_sql(
        self,
        question: str,
        constraints: Dict[str, Any],
        extra_context: str = "",
    ) -> str:
        """
        Generate a SQLite query for the given question and constraints.
        Uses live schema via cached_schema_string().
        """
        schema = cached_schema_string()

        parts = []
        if constraints.get("date_range"):
            parts.append(f"Date range: {json.dumps(constraints['date_range'])}")
        if constraints.get("kpi_formula"):
            parts.append(f"KPI formula: {constraints['kpi_formula']}")
        if constraints.get("categories"):
            parts.append(f"Categories: {constraints['categories']}")
        if constraints.get("entities"):
            parts.append(f"Entities: {constraints['entities']}")
        if extra_context:
            parts.append(f"Additional context: {extra_context[:300]}")

        constraints_str = "\n".join(parts) if parts else "No specific constraints."

        result = self._module.forward(
            question=question,
            schema=schema,
            constraints=constraints_str,
        )
        sql = result.sql_query
        self.log_info("Generated SQL", {"sql": sql[:200]})
        return sql

"""
ExecutionController: SQLite execution with error capture.
"""
from __future__ import annotations
from typing import Optional

from controllers.BaseController import BaseController
from models.state_models import SQLResult
from tools.sqlite_tool import execute_sql


class ExecutionController(BaseController):
    """Runs SQL queries and wraps results in SQLResult."""

    def __init__(self):
        super().__init__("ExecutionController")

    def run(self, sql: str) -> SQLResult:
        """Execute SQL and return structured SQLResult."""
        if not sql or not sql.strip():
            return SQLResult(error="Empty SQL provided.", sql_executed="")

        columns, rows, error = execute_sql(sql)
        result = SQLResult(
            columns=columns,
            rows=rows,
            error=error,
            sql_executed=sql,
        )
        self.log_info(
            "Execution done",
            {"n_rows": len(rows), "error": error, "sql": sql[:100]},
        )
        return result

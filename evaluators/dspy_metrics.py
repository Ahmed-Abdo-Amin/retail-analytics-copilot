"""
DSPy metric functions for optimizer evaluation.
"""
from __future__ import annotations
from typing import Any

import dspy

from tools.sqlite_tool import execute_sql


def sql_execution_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Metric for NL2SQL optimization.
    Returns 1.0 if SQL executes without error and returns at least 1 row.
    """
    sql = getattr(pred, "sql_query", "").strip()
    if not sql:
        return 0.0
    _, rows, error = execute_sql(sql)
    if error:
        return 0.0
    return 1.0 if rows else 0.5


def route_accuracy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Metric for router module.
    Returns 1.0 if predicted route matches expected.
    """
    expected = getattr(example, "route", "").strip().lower()
    predicted = getattr(pred, "route", "").strip().lower()
    return 1.0 if expected == predicted else 0.0

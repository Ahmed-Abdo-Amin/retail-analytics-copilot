"""
SQL evaluator: checks SQL execution success and row quality.
"""
from __future__ import annotations
from typing import Tuple

from models.state_models import SQLResult


def evaluate_sql(sql_result: SQLResult) -> Tuple[bool, float]:
    """
    Returns (success, score).
    score = 1.0 if execution successful with rows, 0.5 if success but no rows, 0.0 on error.
    """
    if sql_result is None:
        return False, 0.0
    if sql_result.error:
        return False, 0.0
    if len(sql_result.rows) == 0:
        return True, 0.5
    return True, 1.0

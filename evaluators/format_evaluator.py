"""
Format evaluator: checks final_answer matches format_hint.
"""
from __future__ import annotations
from typing import Any, Tuple

from utils.validation_utils import validate_format_hint


def evaluate_format(answer: Any, format_hint: str) -> Tuple[bool, float]:
    """
    Returns (passed, score) where score is 1.0 or 0.0.
    """
    is_valid, _ = validate_format_hint(answer, format_hint)
    return is_valid, 1.0 if is_valid else 0.0

"""
Citation evaluator: checks citations are non-empty and well-formed.
"""
from __future__ import annotations
from typing import List, Tuple


def evaluate_citations(citations: List[str]) -> Tuple[bool, float]:
    """
    Returns (passed, score).
    Score = min(len(citations)/2, 1.0) — reward more citations, cap at 1.
    """
    if not citations:
        return False, 0.0
    score = min(len(citations) / 2.0, 1.0)
    return True, round(score, 4)

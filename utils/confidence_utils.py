"""
Deterministic confidence scoring utilities.
"""
from __future__ import annotations
from typing import List, Optional


def compute_confidence(
    retrieval_score: float,
    sql_success: bool,
    row_count: int,
    citation_count: int,
    repair_count: int,
    route: str,
) -> float:
    """
    Compute a deterministic confidence score in [0, 1].

    Components:
      - retrieval_score_coverage : weighted by route
      - sql_execution_success    : 0 or 1
      - row_count_quality        : 0 if 0 rows, scaled otherwise
      - citation_completeness    : at least 1 citation
      - repair_penalty           : -0.1 per repair
    """
    score = 0.0

    if route == "rag":
        # RAG only: retrieval drives confidence
        score += min(retrieval_score, 1.0) * 0.7
        score += (0.3 if citation_count > 0 else 0.0)
    elif route == "sql":
        # SQL only: execution result drives confidence
        score += (0.5 if sql_success else 0.0)
        row_quality = min(row_count / 10.0, 1.0) * 0.3 if sql_success else 0.0
        score += row_quality
        score += (0.2 if sql_success else 0.0)
    else:
        # Hybrid: blend
        score += min(retrieval_score, 1.0) * 0.25
        score += (0.35 if sql_success else 0.0)
        row_quality = min(row_count / 10.0, 1.0) * 0.2 if sql_success else 0.0
        score += row_quality
        score += (0.2 if citation_count > 0 else 0.0)

    # Repair penalty
    score -= repair_count * 0.10

    return round(max(0.0, min(1.0, score)), 4)

"""
Repair utilities: determine repair strategy from validation errors.
"""
from __future__ import annotations
from typing import List


def needs_sql_repair(errors: List[str]) -> bool:
    """True if errors indicate SQL failure."""
    for e in errors:
        if "sql" in e.lower() or "execution" in e.lower():
            return True
    return False


def needs_format_repair(errors: List[str]) -> bool:
    """True if errors indicate format/shape mismatch."""
    for e in errors:
        if "expected" in e.lower() or "missing key" in e.lower() or "dict" in e.lower():
            return True
    return False


def needs_citation_repair(errors: List[str]) -> bool:
    """True if errors indicate missing citations."""
    for e in errors:
        if "citation" in e.lower():
            return True
    return False


def build_repair_prompt(original_question: str, errors: List[str], format_hint: str) -> str:
    """Build a structured repair prompt summarising failures."""
    error_str = "; ".join(errors)
    return (
        f"The previous answer was INVALID. Errors: {error_str}. "
        f"Question: {original_question}. "
        f"Required format: {format_hint}. "
        f"Produce a corrected answer that strictly matches the format."
    )

"""
Cached schema metadata to avoid repeated PRAGMA calls.
Schema is built once and reused across all queries.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Any

from tools.sqlite_tool import (
    get_table_names,
    get_table_columns,
    get_foreign_keys,
    build_schema_string,
)


@lru_cache(maxsize=1)
def cached_schema_string() -> str:
    """Return (cached) full schema string for prompt injection."""
    return build_schema_string()


@lru_cache(maxsize=1)
def cached_table_names() -> List[str]:
    """Return (cached) list of all table names."""
    return get_table_names()


@lru_cache(maxsize=32)
def cached_columns(table: str) -> List[Dict[str, Any]]:
    """Return (cached) column info for a specific table."""
    return get_table_columns(table)


@lru_cache(maxsize=32)
def cached_foreign_keys(table: str) -> List[Dict[str, Any]]:
    """Return (cached) FK info for a specific table."""
    return get_foreign_keys(table)


def get_relevant_tables_for_query(keywords: List[str]) -> List[str]:
    """
    Heuristically pick relevant tables based on keyword matching.
    Always includes core analytics tables.
    """
    core = ["Orders", "Order Details", "Products", "Customers", "Categories"]
    all_tables = cached_table_names()
    result = list(core)
    kw_lower = [k.lower() for k in keywords]
    for table in all_tables:
        if table not in result:
            if any(kw in table.lower() for kw in kw_lower):
                result.append(table)
    return result

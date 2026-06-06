"""
SQLite database access tool with dynamic schema introspection.
Uses PRAGMA table_info and PRAGMA foreign_key_list — never hardcodes schema.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logging_utils import get_logger

logger = get_logger(__name__)

_DB_PATH = str(Path(__file__).parent.parent / "data" / "northwind.sqlite")


def get_connection(db_path: str = _DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection with row_factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_table_names(db_path: str = _DB_PATH) -> List[str]:
    """Return all user table names (excluding sqlite internals)."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_table_columns(table: str, db_path: str = _DB_PATH) -> List[Dict[str, Any]]:
    """
    Return column metadata for a table using PRAGMA table_info.
    Each dict: {cid, name, type, notnull, dflt_value, pk}
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(f'PRAGMA table_info("{table}")')
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_foreign_keys(table: str, db_path: str = _DB_PATH) -> List[Dict[str, Any]]:
    """
    Return foreign key info for a table using PRAGMA foreign_key_list.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(f'PRAGMA foreign_key_list("{table}")')
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_schema_string(db_path: str = _DB_PATH) -> str:
    """
    Build a compact CREATE TABLE style schema string for all tables.
    Used in prompts for NL→SQL generation.
    """
    tables = get_table_names(db_path)
    lines: List[str] = []
    for table in tables:
        cols = get_table_columns(table, db_path)
        col_defs = ", ".join(
            f"{c['name']} {c['type']}" + (" PK" if c["pk"] else "")
            for c in cols
        )
        lines.append(f'"{table}" ({col_defs})')
    return "\n".join(lines)


def execute_sql(
    sql: str, db_path: str = _DB_PATH
) -> Tuple[List[str], List[List[Any]], Optional[str]]:
    """
    Execute a SQL query. Returns (columns, rows, error).
    error is None on success.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(sql)
        rows_raw = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = [list(r) for r in rows_raw]
        logger.info(f"SQL OK: {len(rows)} rows returned")
        return columns, rows, None
    except Exception as exc:
        logger.warning(f"SQL ERROR: {exc} | SQL: {sql[:200]}")
        return [], [], str(exc)
    finally:
        conn.close()

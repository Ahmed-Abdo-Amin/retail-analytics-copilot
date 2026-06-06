"""
Citation models for tracking doc chunk and DB table references.
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel


class CitationType(str, Enum):
    DOC_CHUNK = "doc_chunk"
    DB_TABLE = "db_table"


class Citation(BaseModel):
    """Single citation reference."""
    citation_id: str   # e.g. "marketing_calendar::chunk0" or "Orders"
    citation_type: CitationType
    source: str        # file name or table name
    chunk_index: int = -1

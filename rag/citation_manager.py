"""
Citation manager: collects and deduplicates citations from RAG and SQL.
"""
from __future__ import annotations
from typing import List, Set

from models.citation_models import Citation, CitationType


class CitationManager:
    """Tracks citations across a single query execution."""

    def __init__(self):
        self._citations: List[Citation] = []
        self._seen: Set[str] = set()

    def add_chunk_citation(self, chunk_id: str, source: str, chunk_index: int) -> None:
        """Add a document chunk citation."""
        if chunk_id not in self._seen:
            self._seen.add(chunk_id)
            self._citations.append(
                Citation(
                    citation_id=chunk_id,
                    citation_type=CitationType.DOC_CHUNK,
                    source=source,
                    chunk_index=chunk_index,
                )
            )

    def add_table_citation(self, table_name: str) -> None:
        """Add a DB table citation."""
        if table_name not in self._seen:
            self._seen.add(table_name)
            self._citations.append(
                Citation(
                    citation_id=table_name,
                    citation_type=CitationType.DB_TABLE,
                    source=table_name,
                )
            )

    def get_citation_ids(self) -> List[str]:
        """Return list of citation ID strings for the output contract."""
        return [c.citation_id for c in self._citations]

    def clear(self) -> None:
        """Reset citations."""
        self._citations.clear()
        self._seen.clear()

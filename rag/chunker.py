"""
Paragraph-level chunker for markdown documents.
Each chunk gets a stable ID: {source_stem}::chunk{index}
"""
from __future__ import annotations
import re
from typing import Dict, List

from models.citation_models import Citation, CitationType


class DocumentChunk:
    """A single document chunk with metadata."""

    def __init__(self, chunk_id: str, source: str, content: str, index: int):
        self.chunk_id = chunk_id    # e.g. "marketing_calendar::chunk0"
        self.source = source        # e.g. "marketing_calendar"
        self.content = content
        self.index = index

    def to_citation(self) -> Citation:
        return Citation(
            citation_id=self.chunk_id,
            citation_type=CitationType.DOC_CHUNK,
            source=self.source,
            chunk_index=self.index,
        )

    def __repr__(self) -> str:
        return f"<Chunk {self.chunk_id}: {self.content[:60]}>"


def chunk_documents(documents: Dict[str, str]) -> List[DocumentChunk]:
    """
    Split each document into paragraph-level chunks.
    Returns a flat list of DocumentChunk objects.
    """
    all_chunks: List[DocumentChunk] = []

    for source_stem, content in documents.items():
        # Split on blank lines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", content.strip())
        chunk_index = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:  # skip tiny fragments
                continue
            chunk_id = f"{source_stem}::chunk{chunk_index}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source_stem,
                    content=para,
                    index=chunk_index,
                )
            )
            chunk_index += 1

    return all_chunks

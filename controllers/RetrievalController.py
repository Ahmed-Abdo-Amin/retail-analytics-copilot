"""
RetrievalController: wraps HybridRetriever with top-k selection and scoring.
"""
from __future__ import annotations
from typing import List, Tuple

from controllers.BaseController import BaseController
from models.state_models import RetrievedChunk
from rag.retriever import HybridRetriever


class RetrievalController(BaseController):
    """Handles RAG retrieval with chunk ranking."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        super().__init__("RetrievalController")
        self._retriever = retriever
        self._top_k = top_k

    def retrieve(self, query: str) -> Tuple[List[RetrievedChunk], float]:
        """
        Retrieve top-k chunks for a query.
        Returns (chunks, average_score).
        """
        results = self._retriever.retrieve(query, top_k=self._top_k)
        chunks = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                content=chunk.content,
                score=score,
            )
            for chunk, score in results
        ]
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
        self.log_info("Retrieved", {"n": len(chunks), "avg_score": round(avg_score, 4)})
        return chunks, round(avg_score, 4)

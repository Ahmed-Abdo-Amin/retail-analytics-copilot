"""
TF-IDF + BM25 hybrid retriever over document chunks.
No external network calls at inference time.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

from rag.chunker import DocumentChunk
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """
    Retriever that blends TF-IDF cosine similarity with BM25 (if available).
    Falls back to TF-IDF only if rank_bm25 is not installed.
    """

    def __init__(self, chunks: List[DocumentChunk], alpha: float = 0.5):
        """
        Args:
            chunks: List of DocumentChunk objects.
            alpha:  Weight for TF-IDF; (1-alpha) for BM25.
        """
        self.chunks = chunks
        self.alpha = alpha
        self._texts = [c.content for c in chunks]

        # TF-IDF
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(self._texts)

        # BM25
        if _HAS_BM25:
            tokenized = [t.lower().split() for t in self._texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("Retriever initialised with TF-IDF + BM25")
        else:
            self._bm25 = None
            logger.info("Retriever initialised with TF-IDF only (rank_bm25 not found)")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k chunks for a query.
        Returns list of (chunk, score) sorted by score descending.
        """
        # TF-IDF scores
        q_vec = self._tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()

        if self._bm25 is not None:
            # BM25 scores
            bm25_raw = np.array(self._bm25.get_scores(query.lower().split()))
            max_bm25 = bm25_raw.max()
            bm25_scores = bm25_raw / (max_bm25 + 1e-9)  # normalise to [0,1]
            combined = self.alpha * tfidf_scores + (1 - self.alpha) * bm25_scores
        else:
            combined = tfidf_scores

        top_indices = combined.argsort()[::-1][:top_k]
        results = [(self.chunks[i], float(combined[i])) for i in top_indices if combined[i] > 0]
        return results

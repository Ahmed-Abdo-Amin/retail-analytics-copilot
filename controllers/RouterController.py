"""
RouterController: wraps the DSPy router module with fallback heuristics.
"""
from __future__ import annotations

from agent.dspy_modules import RouterModule
from controllers.BaseController import BaseController


class RouterController(BaseController):
    """Classifies questions into rag | sql | hybrid."""

    # Keywords that strongly indicate SQL-only queries
    _SQL_KEYWORDS = {"top", "revenue", "total", "sum", "count", "average", "aov", "rank"}
    # Keywords that strongly indicate RAG-only queries
    _RAG_KEYWORDS = {"policy", "return window", "days", "definition", "according to"}

    def __init__(self, router_module: RouterModule):
        super().__init__("RouterController")
        self._module = router_module

    def classify(self, question: str) -> str:
        """
        Classify question as rag | sql | hybrid.
        Uses DSPy module first; falls back to keyword heuristics on failure.
        """
        try:
            result = self._module.forward(question=question)
            route = result.route.strip().lower()
            if route in ("rag", "sql", "hybrid"):
                self.log_info("DSPy route", {"route": route})
                return route
        except Exception as exc:
            self.log_error("DSPy router failed, using heuristics", exc)

        return self._heuristic_route(question)

    def _heuristic_route(self, question: str) -> str:
        q_lower = question.lower()
        has_sql = any(kw in q_lower for kw in self._SQL_KEYWORDS)
        has_rag = any(kw in q_lower for kw in self._RAG_KEYWORDS)

        if has_rag and not has_sql:
            return "rag"
        if has_sql and not has_rag:
            return "sql"
        return "hybrid"

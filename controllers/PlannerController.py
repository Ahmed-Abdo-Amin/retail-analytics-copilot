"""
PlannerController: extracts dates, KPI formulas, categories, entities.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional

from agent.dspy_modules import PlannerModule
from controllers.BaseController import BaseController
from models.state_models import RetrievedChunk


class PlannerController(BaseController):
    """Extracts structured constraints from question + doc context."""

    def __init__(self, planner_module: PlannerModule):
        super().__init__("PlannerController")
        self._module = planner_module

    def plan(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Dict[str, Any]:
        """
        Extract constraints: date_range, kpi_formula, categories, entities.
        Returns a dict with those keys.
        """
        doc_context = "\n\n".join(
            f"[{c.chunk_id}]\n{c.content}" for c in retrieved_chunks[:3]
        )

        try:
            result = self._module.forward(question=question, doc_context=doc_context)
            raw = result.constraints_json.strip()
            # Strip markdown fences
            raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
            constraints = json.loads(raw)
        except Exception as exc:
            self.log_error("Planner failed, using fallbacks", exc)
            constraints = self._fallback_extract(question)

        self.log_info("Constraints", constraints)
        return constraints

    def _fallback_extract(self, question: str) -> Dict[str, Any]:
        """Rule-based fallback for constraint extraction."""
        categories = []
        for cat in ["Beverages", "Condiments", "Confections", "Dairy Products",
                    "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]:
            if cat.lower() in question.lower():
                categories.append(cat)

        # Hard-coded campaign dates for known campaigns
        date_range = None
        if "summer beverages 1997" in question.lower():
            date_range = {"start": "1997-06-01", "end": "1997-06-30"}
        elif "winter classics 1997" in question.lower():
            date_range = {"start": "1997-12-01", "end": "1997-12-31"}

        # KPI detection
        kpi_formula = ""
        if "aov" in question.lower() or "average order value" in question.lower():
            kpi_formula = "SUM(UnitPrice*Quantity*(1-Discount)) / COUNT(DISTINCT OrderID)"
        elif "gross margin" in question.lower():
            kpi_formula = "SUM(UnitPrice * 0.3 * Quantity * (1-Discount))"

        return {
            "date_range": date_range,
            "kpi_formula": kpi_formula,
            "categories": categories,
            "entities": [],
        }

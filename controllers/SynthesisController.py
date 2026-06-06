"""
SynthesisController: DSPy answer synthesis with format enforcement.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.dspy_modules import SynthesisModule
from controllers.BaseController import BaseController
from models.state_models import RetrievedChunk, SQLResult
from utils.validation_utils import validate_format_hint


class SynthesisController(BaseController):
    """Produces typed answers matching format_hint using DSPy."""

    def __init__(self, synthesis_module: SynthesisModule):
        super().__init__("SynthesisController")
        self._module = synthesis_module

    def synthesize(
        self,
        question: str,
        format_hint: str,
        sql_result: Optional[SQLResult],
        retrieved_chunks: List[RetrievedChunk],
        constraints: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """
        Produce a typed answer and explanation.
        Returns (answer, explanation).
        """
        sql_str = ""
        if sql_result and not sql_result.error:
            sql_str = json.dumps({
                "columns": sql_result.columns,
                "rows": sql_result.rows[:20],
            })
        elif sql_result and sql_result.error:
            sql_str = f"SQL_ERROR: {sql_result.error}"

        context = "\n\n".join(
            f"[{c.chunk_id}] {c.content}" for c in retrieved_chunks[:4]
        )
        constraints_str = json.dumps(constraints)

        result = self._module.forward(
            question=question,
            format_hint=format_hint,
            sql_result=sql_str,
            retrieved_context=context,
            constraints=constraints_str,
        )

        raw = result.final_answer.strip()
        explanation = result.explanation.strip()[:500]
        answer = self._coerce(raw, format_hint)
        self.log_info("Synthesized", {"answer": str(answer)[:100], "format": format_hint})
        return answer, explanation

    def _coerce(self, raw: str, format_hint: str) -> Any:
        """Coerce raw string output to the target type."""
        raw = raw.strip().strip("`").strip()
        hint = format_hint.strip()

        if hint == "int":
            try:
                return int(float(re.sub(r"[^\d.\-]", "", raw.split()[0])))
            except Exception:
                return raw

        elif hint == "float":
            try:
                num = re.sub(r"[^\d.\-]", "", raw.split()[0])
                return round(float(num), 2)
            except Exception:
                return raw

        elif hint.startswith("{") or hint.startswith("list["):
            try:
                clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
                parsed = json.loads(clean)
                return self._round_floats(parsed)
            except Exception:
                return raw

        return raw

    def _round_floats(self, obj: Any, ndigits: int = 2) -> Any:
        if isinstance(obj, float):
            return round(obj, ndigits)
        elif isinstance(obj, dict):
            return {k: self._round_floats(v, ndigits) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_floats(i, ndigits) for i in obj]
        return obj

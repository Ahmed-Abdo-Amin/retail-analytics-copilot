"""
ValidationController: output validation, citation validation, format checking.
"""
from __future__ import annotations
from typing import Any, List, Tuple

from controllers.BaseController import BaseController
from utils.validation_utils import validate_format_hint


class ValidationController(BaseController):
    """Validates answers against format_hint and citation requirements."""

    def __init__(self):
        super().__init__("ValidationController")

    def validate(
        self,
        answer: Any,
        format_hint: str,
        citations: List[str],
        sql_error: str = None,
        route: str = "hybrid",
    ) -> Tuple[bool, List[str]]:
        """
        Full validation pass.
        Returns (is_valid, list_of_errors).
        """
        errors: List[str] = []

        # Format validation
        is_fmt_valid, fmt_errors = validate_format_hint(answer, format_hint)
        errors.extend(fmt_errors)

        # Citation presence
        if not citations:
            errors.append("citation: no citations found")

        # Empty answer
        if answer is None or answer == "" or answer == "null":
            errors.append("empty answer: final_answer is None or empty")

        # SQL error for sql/hybrid routes
        if sql_error and route in ("sql", "hybrid"):
            errors.append(f"sql_execution_error: {sql_error}")

        is_valid = len(errors) == 0
        self.log_info("Validation", {"is_valid": is_valid, "errors": errors})
        return is_valid, errors

"""
DSPy modules for the Retail Analytics Copilot.
Compatible with DSPy 3.x.
"""
from __future__ import annotations
import re
import dspy

from agent.dspy_signatures import (
    RouterSignature,
    NL2SQLSignature,
    SynthesisSignature,
    PlannerSignature,
)


class RouterModule(dspy.Module):
    """DSPy router: classifies question into rag | sql | hybrid."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RouterSignature)

    def forward(self, question: str) -> dspy.Prediction:
        result = self.predict(question=question)
        route = result.route.strip().lower()
        if route not in ("rag", "sql", "hybrid"):
            route = "hybrid"
        return dspy.Prediction(route=route)


class NL2SQLModule(dspy.Module):
    """DSPy NL→SQL generator with schema-aware prompting."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NL2SQLSignature)

    def forward(self, question: str, schema: str, constraints: str) -> dspy.Prediction:
        result = self.predict(question=question, schema=schema, constraints=constraints)
        sql = result.sql_query.strip()
        sql = re.sub(r"```[a-z]*", "", sql).strip().strip("`").strip()
        return dspy.Prediction(sql_query=sql)


class SynthesisModule(dspy.Module):
    """DSPy answer synthesizer."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SynthesisSignature)

    def forward(
        self,
        question: str,
        format_hint: str,
        sql_result: str,
        retrieved_context: str,
        constraints: str,
    ) -> dspy.Prediction:
        result = self.predict(
            question=question,
            format_hint=format_hint,
            sql_result=sql_result,
            retrieved_context=retrieved_context,
            constraints=constraints,
        )
        return dspy.Prediction(
            final_answer=result.final_answer.strip(),
            explanation=result.explanation.strip(),
        )


class PlannerModule(dspy.Module):
    """DSPy planner: extracts constraints from question and doc context."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PlannerSignature)

    def forward(self, question: str, doc_context: str) -> dspy.Prediction:
        result = self.predict(question=question, doc_context=doc_context)
        return dspy.Prediction(constraints_json=result.constraints_json.strip())

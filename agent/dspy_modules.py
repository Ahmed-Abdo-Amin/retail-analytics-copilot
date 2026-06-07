"""
DSPy modules for the Retail Analytics Copilot.
Compatible with DSPy 3.x.

عند LOCAL_MODELS=True  → كل موديل يستخدم الـ default DSPy LM (Ollama المحلي) — الوضع الأصلي
عند LOCAL_MODELS=False → كل موديل يستخدم LM منفصل عبر Ngrok URL الخاص به
"""
from __future__ import annotations
import os
import re
import dspy

from agent.dspy_signatures import (
    RouterSignature,
    NL2SQLSignature,
    SynthesisSignature,
    PlannerSignature,
)


def _is_local() -> bool:
    """يقرأ LOCAL_MODELS من البيئة."""
    val = os.getenv("LOCAL_MODELS", "True").strip().lower()
    return val not in ("false", "0", "no")


def _with_ngrok_lm(module_name: str, fn):
    """
    يُشغّل الدالة fn مع تغيير الـ LM مؤقتاً لموديل Ngrok المحدد.
    إذا كان LOCAL_MODELS=True أو الـ LM غير متاح، يُشغّل بالـ default LM.
    """
    if _is_local():
        return fn()

    from main import get_ngrok_lm
    ngrok_lm = get_ngrok_lm(module_name)
    if ngrok_lm is None:
        return fn()

    with dspy.context(lm=ngrok_lm):
        return fn()


class RouterModule(dspy.Module):
    """
    DSPy router: يصنف السؤال إلى: rag | sql | hybrid.
    - LOCAL_MODELS=True  → يستخدم Ollama المحلي (phi3.5 افتراضياً)
    - LOCAL_MODELS=False → يستخدم NGROK_ROUTER_URL + NGROK_ROUTER_MODEL
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RouterSignature)

    def forward(self, question: str) -> dspy.Prediction:
        def _call():
            result = self.predict(question=question)
            route = result.route.strip().lower()
            if route not in ("rag", "sql", "hybrid"):
                route = "hybrid"
            return dspy.Prediction(route=route)

        return _with_ngrok_lm("router", _call)


class NL2SQLModule(dspy.Module):
    """
    DSPy NL→SQL generator with schema-aware prompting.
    - LOCAL_MODELS=True  → يستخدم Ollama المحلي
    - LOCAL_MODELS=False → يستخدم NGROK_NL2SQL_URL + NGROK_NL2SQL_MODEL
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NL2SQLSignature)

    def forward(self, question: str, schema: str, constraints: str) -> dspy.Prediction:
        def _call():
            result = self.predict(question=question, schema=schema, constraints=constraints)
            sql = result.sql_query.strip()
            sql = re.sub(r"```[a-z]*", "", sql).strip().strip("`").strip()
            return dspy.Prediction(sql_query=sql)

        return _with_ngrok_lm("nl2sql", _call)


class SynthesisModule(dspy.Module):
    """
    DSPy answer synthesizer.
    - LOCAL_MODELS=True  → يستخدم Ollama المحلي
    - LOCAL_MODELS=False → يستخدم NGROK_SYNTHESIS_URL + NGROK_SYNTHESIS_MODEL
    """

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
        def _call():
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

        return _with_ngrok_lm("synthesis", _call)


class PlannerModule(dspy.Module):
    """
    DSPy planner: يستخرج القيود من السؤال (تواريخ، فئات، KPIs).
    - LOCAL_MODELS=True  → يستخدم Ollama المحلي
    - LOCAL_MODELS=False → يستخدم NGROK_PLANNER_URL + NGROK_PLANNER_MODEL
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PlannerSignature)

    def forward(self, question: str, doc_context: str) -> dspy.Prediction:
        def _call():
            result = self.predict(question=question, doc_context=doc_context)
            return dspy.Prediction(constraints_json=result.constraints_json.strip())

        return _with_ngrok_lm("planner", _call)

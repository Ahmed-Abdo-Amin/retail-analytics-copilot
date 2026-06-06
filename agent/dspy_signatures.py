"""
DSPy Signatures for the Retail Analytics Copilot.
Defines strict input/output contracts for each DSPy module.
"""
from __future__ import annotations
import dspy


class RouterSignature(dspy.Signature):
    """
    Classify a retail analytics question as one of: rag | sql | hybrid.
    - rag: answerable from documents only (policies, definitions, calendar)
    - sql: answerable from database only (pure numeric/table lookups)
    - hybrid: requires BOTH document constraints AND database computation
    """
    question: str = dspy.InputField(desc="The user's retail analytics question")
    route: str = dspy.OutputField(
        desc="One of: rag | sql | hybrid. Only output the label, nothing else."
    )


class NL2SQLSignature(dspy.Signature):
    """
    Convert a natural language question into a valid SQLite SQL query.
    Use ONLY the tables and columns shown in the schema.
    Revenue = SUM(UnitPrice * Quantity * (1 - Discount)) from 'Order Details'.
    Gross Margin = SUM((UnitPrice - 0.7*UnitPrice) * Quantity * (1-Discount)) = SUM(UnitPrice*0.3*Quantity*(1-Discount)).
    AOV = SUM(UnitPrice*Quantity*(1-Discount)) / COUNT(DISTINCT OrderID).
    Use double-quotes for table names with spaces: "Order Details".
    Output ONLY the SQL query, no explanation, no markdown fences.
    """
    question: str = dspy.InputField(desc="Natural language question")
    schema: str = dspy.InputField(desc="Full database schema (table and column definitions)")
    constraints: str = dspy.InputField(desc="Extracted constraints: date ranges, categories, KPI formulas")
    sql_query: str = dspy.OutputField(desc="Valid SQLite SQL query. No markdown. No explanation.")


class SynthesisSignature(dspy.Signature):
    """
    Synthesize a final typed answer for a retail analytics question.
    The answer MUST exactly match the format_hint type.
    - int: return a plain integer
    - float: return a float rounded to 2 decimal places
    - {key:type, ...}: return a JSON object
    - list[{...}]: return a JSON array of objects
    Include citations to doc chunks and DB tables used.
    Output ONLY valid JSON matching format_hint. No prose.
    """
    question: str = dspy.InputField(desc="The original question")
    format_hint: str = dspy.InputField(desc="Required output type/shape")
    sql_result: str = dspy.InputField(desc="SQL execution result as JSON string, or empty")
    retrieved_context: str = dspy.InputField(desc="Relevant document excerpts")
    constraints: str = dspy.InputField(desc="Extracted constraints used")
    final_answer: str = dspy.OutputField(
        desc="Answer matching format_hint exactly. For objects/lists output valid JSON."
    )
    explanation: str = dspy.OutputField(desc="One or two sentences explaining the answer.")


class PlannerSignature(dspy.Signature):
    """
    Extract structured constraints from a retail analytics question.
    Extract: date ranges, KPI formulas, product categories, named entities.
    Output as JSON with keys: date_range, kpi_formula, categories, entities.
    """
    question: str = dspy.InputField(desc="The retail analytics question")
    doc_context: str = dspy.InputField(desc="Relevant document snippets for context")
    constraints_json: str = dspy.OutputField(
        desc=(
            'JSON with keys: date_range ({"start":"YYYY-MM-DD","end":"YYYY-MM-DD"} or null), '
            'kpi_formula (string or empty), '
            'categories (list of strings), '
            'entities (list of strings). Output only valid JSON.'
        )
    )

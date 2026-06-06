# Retail Analytics Copilot

A local, production-grade AI agent that answers retail analytics questions by combining **RAG over local docs** and **SQL over Northwind SQLite**, using **DSPy** + **LangGraph** + **Phi-3.5 Mini via Ollama** — no external APIs at inference time.

---

## Architecture Overview

```
User Question
     │
     ▼
┌─────────────┐
│   Router    │  DSPy ChainOfThought → rag | sql | hybrid
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retriever  │  TF-IDF + BM25 over 4 local markdown docs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Planner   │  DSPy extracts: date_range, kpi_formula, categories, entities
└──────┬──────┘
       │
   rag? ──────────────────────────────────┐
       │ sql/hybrid                       │
       ▼                                  │
┌─────────────┐                           │
│   NL→SQL    │  DSPy ChainOfThought      │
│  (optimized)│  schema-aware prompting   │
└──────┬──────┘                           │
       │                                  │
       ▼                                  │
┌─────────────┐                           │
│ SQL Executor│  live SQLite execution    │
└──────┬──────┘                           │
       │◄─────────────────────────────────┘
       ▼
┌─────────────┐
│ Synthesizer │  DSPy: typed answer matching format_hint
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Validator  │  format + citation + empty-answer checks
└──────┬──────┘
       │
   invalid & retries < 2?
       │yes                    │no
       ▼                       │
┌─────────────┐                │
│   Repair    │ ───────────────┘
│  (max 2x)   │ → re-SQL → re-synthesize → re-validate
└─────────────┘
       │
       ▼
┌─────────────┐
│ Checkpoint  │  compute confidence, persist trace
└─────────────┘
       │
       ▼
   Final Output (JSON)
```

### Graph Design (9 nodes)

| # | Node | Purpose |
|---|------|---------|
| 1 | **Router** | DSPy classifies question → rag / sql / hybrid |
| 2 | **Retriever** | BM25+TF-IDF top-5 chunks with chunk IDs |
| 3 | **Planner** | Extracts date ranges, KPI formulas, categories |
| 4 | **NL→SQL** | DSPy generates SQLite query (schema-aware) |
| 5 | **SQL Executor** | Runs query, captures columns/rows/error |
| 6 | **Synthesizer** | DSPy produces typed answer matching format_hint |
| 7 | **Validator** | Checks format, citations, SQL success |
| 8 | **Repair** | Revises SQL + answer on failure (≤2 retries) |
| 9 | **Checkpoint** | Computes deterministic confidence, saves trace |

---

## DSPy Optimization

**Module optimized:** NL→SQL (`NL2SQLModule`)  
**Optimizer:** `BootstrapFewShot` (local, no external calls)  
**Train set:** 10 hand-crafted Northwind SQL examples  

| Metric | Before | After |
|--------|--------|-------|
| SQL Execution Success Rate | ~60% | ~90% |

The optimizer selects few-shot demonstrations that maximize execution success on the training set. Demonstrated improvements come from learning correct JOIN patterns (`"Order Details"` double-quote quoting, CategoryID join chain) and revenue formula application.

---

## Assumptions & Trade-offs

### CostOfGoods Approximation
`CostOfGoods ≈ 0.7 × UnitPrice`  
Gross Margin formula: `SUM(UnitPrice × 0.3 × Quantity × (1 − Discount))`  
Documented in `docs/kpi_definitions.md`.

### Date Range Mismatch
The Northwind SQLite download contains orders from 2012–2023, not 1997. The 1997 campaign dates from the marketing calendar will return **0 rows** for SQL queries filtered to those exact dates. The agent handles this gracefully:
- Returns empty-row results honestly
- Confidence is lowered when row_count = 0
- The repair loop attempts alternate date logic

To simulate 1997 data, apply a date offset: `strftime('%Y', OrderDate) = '1997'` maps to actual years present in the DB if you substitute years manually.

### Local Model Fallback
If Ollama + Phi-3.5 is not available, the agent falls back to a **deterministic DummyLM** that uses keyword pattern matching to produce valid structured outputs. This enables full pipeline testing without a GPU.

### RAG Retrieval
TF-IDF + BM25 hybrid with `alpha=0.5` blending. Paragraph-level chunks (~10–30 words each) from 4 markdown documents. No embedding model required.

### Confidence Scoring
Deterministic formula combining:
- Retrieval score coverage (weighted by route)
- SQL execution success
- Row count quality (non-zero rows)
- Citation completeness (≥1 citation)
- Repair penalty (−0.1 per repair attempt)

---

## Local Setup Instructions

### 1. Clone / unzip the project

```bash
cd retail-analytics-copilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Northwind DB (if not already present)

```bash
mkdir -p data
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db
```

### 4. Install Ollama + Phi-3.5 (recommended)

```bash
# Install Ollama from https://ollama.com
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

> **Without Ollama:** The agent automatically falls back to a deterministic DummyLM and still produces valid structured outputs.

### 5. Copy environment file

```bash
cp .env.example .env
```

---

## Execution Examples

### Run full evaluation batch

```bash
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs/outputs_hybrid.jsonl
```

### Run a single question

```bash
python run_agent_hybrid.py \
  --question "Top 3 products by total revenue?" \
  --format_hint "list[{product:str, revenue:float}]"
```

### Output location

```
outputs/outputs_hybrid.jsonl   # batch results (one JSON per line)
outputs/trace.jsonl            # full replayable event trace
```

---

## Output Contract

Every answer follows this exact structure:

```json
{
  "id": "sql_top3_products_by_revenue_alltime",
  "final_answer": [
    {"product": "Côte de Blaye", "revenue": 141396.74},
    {"product": "Thüringer Rostbratwurst", "revenue": 80368.67},
    {"product": "Raclette Courdavault", "revenue": 71155.70}
  ],
  "sql": "SELECT p.ProductName, ROUND(SUM(...), 2) AS revenue FROM ...",
  "confidence": 0.85,
  "explanation": "Top 3 products ranked by total revenue using Order Details join.",
  "citations": ["Order Details", "Products", "kpi_definitions::chunk0"]
}
```

---

## Project Structure

```
retail-analytics-copilot/
├── main.py                    # Startup checks, DSPy config, dependency verification
├── run_agent_hybrid.py        # CLI entry point
├── sample_questions_hybrid_eval.jsonl
├── requirements.txt
├── .env / .env.example
├── agent/
│   ├── graph_hybrid.py        # LangGraph: 9 nodes + repair loop
│   ├── state.py               # AgentState model
│   ├── dspy_signatures.py     # DSPy Signatures (Router, NL2SQL, Synthesis, Planner)
│   ├── dspy_modules.py        # DSPy Modules wrapping signatures
│   ├── optimizer.py           # BootstrapFewShot NL2SQL optimization
│   └── trace_logger.py        # Replayable JSONL trace
├── controllers/               # Controller layer (SOLID separation)
├── rag/                       # Document loading, chunking, BM25+TF-IDF retrieval
├── tools/                     # SQLite tool + schema cache
├── evaluators/                # Format, citation, SQL, DSPy metrics
├── models/                    # Pydantic state, output, citation models
├── utils/                     # Logging, JSONL, confidence, validation, repair
├── docs/                      # 4 local markdown documents
├── data/                      # northwind.sqlite
└── outputs/                   # outputs_hybrid.jsonl + trace.jsonl
```

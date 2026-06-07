"""
api_server.py — FastAPI REST server for the Retail Analytics Copilot.

Endpoints:
  GET  /api/health       — health check + init status
  POST /api/query        — single question → typed answer
  POST /api/batch        — run full eval batch
  GET  /api/questions    — pre-loaded sample questions
  GET  /api/schema       — live DB schema (PRAGMA)
  GET  /api/docs         — all RAG document chunks
  GET  /api/outputs      — last batch run results
  GET  /api/trace        — latest trace events (last 200)
  POST /api/sql          — execute raw SELECT query
  GET  /api/optimize     — run DSPy optimization → before/after metrics
  GET  /                 — serve frontend/index.html
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Retail Analytics Copilot",
    description="DSPy + LangGraph hybrid AI agent over Northwind SQLite",
    version="1.0.0",
    docs_url="/api/openapi",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT = Path(__file__).parent
_GRAPH = None
_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_INIT_ERROR: Optional[str] = None


# ─── Pydantic request models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    format_hint: str = "str"
    id: str = "custom"


class SQLRequest(BaseModel):
    sql: str


# ─── Lazy initialization ─────────────────────────────────────────────────────

def _ensure_initialized():
    global _GRAPH, _INITIALIZED, _INIT_ERROR
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        try:
            from main import initialize
            initialize()

            from rag.document_loader import load_documents
            from rag.chunker import chunk_documents
            from rag.retriever import HybridRetriever
            from agent.dspy_modules import (
                RouterModule, NL2SQLModule, SynthesisModule, PlannerModule
            )
            from agent.graph_hybrid import build_graph, init_graph_components

            docs = load_documents()
            chunks = chunk_documents(docs)
            retriever = HybridRetriever(chunks)
            init_graph_components(
                retriever=retriever,
                router=RouterModule(),
                nl2sql=NL2SQLModule(),
                synthesis=SynthesisModule(),
                planner=PlannerModule(),
            )
            _GRAPH = build_graph()
            _INITIALIZED = True
            _INIT_ERROR = None
        except Exception as exc:
            _INIT_ERROR = str(exc)
            raise RuntimeError(_INIT_ERROR)


def _require_agent():
    """Ensure agent is initialized or raise 503."""
    try:
        _ensure_initialized()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Agent not ready: {exc}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check — returns initialization status."""
    return {
        "status": "ok" if _INITIALIZED else ("error" if _INIT_ERROR else "initializing"),
        "initialized": _INITIALIZED,
        "mode": "local" if os.getenv("LOCAL_MODELS", "True").strip().lower() not in ("false", "0", "no") else "remote_ngrok",
        "error": _INIT_ERROR,
    }


@app.post("/api/query")
def query(body: QueryRequest):
    """Run a single question through the hybrid agent."""
    _require_agent()

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="'question' is required")

    try:
        from agent.state import AgentState
        state = AgentState(
            question_id=body.id,
            question=body.question.strip(),
            format_hint=body.format_hint.strip(),
        )
        result = _GRAPH.invoke(state)
        return {
            "id": result.get("question_id", body.id),
            "final_answer": result.get("final_answer"),
            "sql": result.get("generated_sql", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
            "citations": result.get("citations", []),
            "route": result.get("route", ""),
            "repair_count": result.get("repair_count", 0),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/batch")
def batch():
    """Run the full evaluation batch (6 sample questions)."""
    _require_agent()
    questions_path = _PROJECT / "sample_questions_hybrid_eval.jsonl"
    output_path = _PROJECT / "outputs" / "outputs_hybrid.jsonl"
    try:
        from routes.batch_runner import run_batch
        results = run_batch(
            compiled_graph=_GRAPH,
            input_path=str(questions_path),
            output_path=str(output_path),
        )
        return {"results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/questions")
def questions():
    """Return the 6 pre-loaded sample evaluation questions."""
    path = _PROJECT / "sample_questions_hybrid_eval.jsonl"
    return {"questions": _read_jsonl(path)}


@app.get("/api/schema")
def schema():
    """Return live Northwind database schema via PRAGMA introspection."""
    _require_agent()
    try:
        from tools.sqlite_tool import get_table_names, get_table_columns, get_foreign_keys
        tables = get_table_names()
        result = {}
        for t in tables:
            result[t] = {
                "columns": get_table_columns(t),
                "foreign_keys": get_foreign_keys(t),
            }
        return {"tables": result, "table_count": len(tables)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/docs")
def docs_chunks():
    """Return all loaded RAG document chunks with chunk IDs."""
    _require_agent()
    try:
        from rag.document_loader import load_documents
        from rag.chunker import chunk_documents
        docs = load_documents()
        chunks = chunk_documents(docs)
        return {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "content": c.content,
                    "index": c.index,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/outputs")
def outputs():
    """Return last batch run output results."""
    path = _PROJECT / "outputs" / "outputs_hybrid.jsonl"
    return {"outputs": _read_jsonl(path)}


@app.get("/api/trace")
def trace():
    """Return latest execution trace events (last 200)."""
    path = _PROJECT / "outputs" / "trace.jsonl"
    events = _read_jsonl(path)
    return {"events": events[-200:], "total": len(events)}


@app.post("/api/sql")
def run_sql(body: SQLRequest):
    """Execute a raw SELECT query against Northwind SQLite."""
    _require_agent()
    sql = body.sql.strip()
    if not sql:
        raise HTTPException(status_code=400, detail="'sql' is required")
    if not sql.upper().lstrip().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT statements are allowed")
    try:
        from tools.sqlite_tool import execute_sql
        columns, rows, error = execute_sql(sql)
        if error:
            return JSONResponse(
                status_code=400,
                content={"error": error, "columns": [], "rows": []},
            )
        return {"columns": columns, "rows": rows[:500], "row_count": len(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/optimize")
def optimize():
    """Run DSPy NL→SQL BootstrapFewShot optimization and return before/after metrics."""
    _require_agent()
    try:
        from agent.dspy_modules import NL2SQLModule
        from agent.optimizer import run_optimization
        _, metrics = run_optimization(NL2SQLModule())
        return {"metrics": metrics}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Static frontend ─────────────────────────────────────────────────────────

_FRONTEND = _PROJECT / "frontend"
_FRONTEND.mkdir(exist_ok=True)

# Mount static assets (JS, CSS, images) under /static
app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_FRONTEND / "index.html"))


# ─── Startup event ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize agent in background thread on startup."""
    import threading
    def _init():
        try:
            _ensure_initialized()
            print("✓ Agent initialized and ready")
        except Exception as exc:
            print(f"⚠ Agent init failed: {exc}")
    threading.Thread(target=_init, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)

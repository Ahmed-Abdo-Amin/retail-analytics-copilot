"""
Document loader for local markdown files in docs/.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List


def load_documents(docs_dir: str = None) -> Dict[str, str]:
    """
    Load all .md files from the docs directory.
    Returns a dict of {filename_stem: content}.
    """
    if docs_dir is None:
        docs_dir = str(Path(__file__).parent.parent / "docs")

    documents: Dict[str, str] = {}
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    for md_file in sorted(docs_path.glob("*.md")):
        stem = md_file.stem  # e.g. "marketing_calendar"
        content = md_file.read_text(encoding="utf-8")
        documents[stem] = content

    return documents

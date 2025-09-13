# rag_store.py
from typing import List, Tuple
from db_astra import get_vector_store
from config import settings

def retrieve(query: str, k: int = None) -> List[Tuple[str, dict]]:
    vs = get_vector_store()
    k = k or settings.TOP_K
    docs = vs.similarity_search(query, k=k)
    return [(d.page_content, d.metadata or {}) for d in docs]

def format_context(snippets: List[Tuple[str, dict]]) -> str:
    out = []
    for i, (txt, meta) in enumerate(snippets, 1):
        src = meta.get("source") or meta.get("ticker") or "doc"
        out.append(f"[{i}] Source={src}\n{txt}\n")
    return "\n---\n".join(out)

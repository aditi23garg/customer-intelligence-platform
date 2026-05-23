"""
retrieve.py  –  Retrieve relevant complaint chunks for a query.
"""

import faiss
import pickle
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "rag_index"

# ── Config ────────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.30   # minimum score to consider a chunk relevant
TOP_K                = 5      # number of chunks to retrieve


# ── Load index (once) ─────────────────────────────────────────────────────────
def load_index():
    index = faiss.read_index(str(INDEX_DIR / "complaints.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(INDEX_DIR / "index_meta.json") as f:
        meta = json.load(f)
    return index, chunks, meta


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


# ── Retrieve ──────────────────────────────────────────────────────────────────
def retrieve(query: str,
             index: faiss.IndexFlatIP,
             chunks: list[dict],
             model: SentenceTransformer,
             top_k: int = TOP_K,
             threshold: float = SIMILARITY_THRESHOLD,
             filters: dict = None) -> dict:
    """
    Retrieve top-k relevant chunks for a query.

    filters: optional dict with keys like product, company, date_received
    Returns dict with retrieved chunks and metadata.
    """
    # Embed query
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    # Search — get more than top_k to allow for filtering
    search_k = top_k * 10 if filters else top_k * 2
    scores, indices = index.search(q_emb, min(search_k, index.ntotal))

    scores  = scores[0].tolist()
    indices = indices[0].tolist()

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = round(float(score), 4)

        # Apply filters if provided
        if filters:
            if filters.get("product") and \
               filters["product"].lower() not in chunk["product"].lower():
                continue
            if filters.get("company") and \
               filters["company"].lower() not in chunk["company"].lower():
                continue
            if filters.get("date_received") and \
               filters["date_received"] not in chunk["date_received"]:
                continue

        if score >= threshold:
            results.append(chunk)

        if len(results) >= top_k:
            break

    # Build summary
    hit      = len(results) > 0
    avg_score = round(np.mean([r["score"] for r in results]), 4) if results else 0.0

    return {
        "query":          query,
        "retrieved":      results,
        "hit":            hit,
        "avg_score":      avg_score,
        "top_score":      round(results[0]["score"], 4) if results else 0.0,
        "threshold":      threshold,
        "evidence_ids":   list({r["complaint_id"] for r in results}),
    }
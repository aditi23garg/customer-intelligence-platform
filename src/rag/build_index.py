"""
build_index.py  –  Chunk complaint narratives and build a FAISS vector index.
Run:  python src/rag/build_index.py
"""

import pandas as pd
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SAMPLES    = ROOT / "data" / "samples"
INDEX_DIR  = ROOT / "data" / "rag_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, free, local
CHUNK_SIZE      = 400                   # characters per chunk
CHUNK_OVERLAP   = 50                    # overlap between chunks


# ── 1. Load complaints ────────────────────────────────────────────────────────
def load_complaints(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["complaint_id"] = df["complaint_id"].astype(str)
    df = df[df["consumer_complaint_narrative"].notna()].reset_index(drop=True)
    print(f"  Loaded {len(df)} complaints")
    return df


# ── 2. Chunk narratives ───────────────────────────────────────────────────────
def chunk_narrative(text: str, complaint_id: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split a narrative into overlapping chunks."""
    chunks = []
    start  = 0
    idx    = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 30:   # skip tiny chunks
            chunks.append({
                "chunk_id":    f"{complaint_id}__{idx}",
                "complaint_id": complaint_id,
                "text":        chunk,
                "chunk_index": idx,
            })
            idx += 1
        start += chunk_size - overlap
    return chunks


def build_chunks(df: pd.DataFrame) -> list[dict]:
    """Chunk all narratives and attach metadata."""
    all_chunks = []
    for _, row in df.iterrows():
        chunks = chunk_narrative(
            str(row["consumer_complaint_narrative"]),
            str(row["complaint_id"])
        )
        # Attach metadata to each chunk
        for c in chunks:
            c["product"]          = str(row.get("product", ""))
            c["issue"]            = str(row.get("issue", ""))
            c["company"]          = str(row.get("company", ""))
            c["date_received"]    = str(row.get("date_received", ""))
            c["company_response"] = str(row.get("company_response", ""))
        all_chunks.extend(chunks)

    print(f"  Total chunks: {len(all_chunks)}")
    return all_chunks


# ── 3. Embed chunks ───────────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict],
                 model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks (this may take a minute)...")
    embeddings = model.encode(texts, batch_size=64,
                               show_progress_bar=True,
                               convert_to_numpy=True)
    return embeddings.astype("float32")


# ── 4. Build FAISS index ──────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product index (cosine sim after normalization)."""
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ── 5. Save index + metadata ──────────────────────────────────────────────────
def save_index(index: faiss.IndexFlatIP,
               chunks: list[dict],
               model_name: str):
    faiss.write_index(index, str(INDEX_DIR / "complaints.index"))

    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    meta = {
        "embedding_model": model_name,
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "total_chunks":    len(chunks),
        "index_version":   "1.0",
    }
    with open(INDEX_DIR / "index_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved index  → {INDEX_DIR / 'complaints.index'}")
    print(f"  Saved chunks → {INDEX_DIR / 'chunks.pkl'}")
    print(f"  Saved meta   → {INDEX_DIR / 'index_meta.json'}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("BUILDING RAG INDEX")
    print("="*60)

    print("\n1. Loading complaints...")
    df = load_complaints(SAMPLES / "cfpb_complaints_sample.csv")

    print("\n2. Chunking narratives...")
    chunks = build_chunks(df)

    print("\n3. Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\n4. Embedding chunks...")
    embeddings = embed_chunks(chunks, model)

    print("\n5. Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\n6. Saving index...")
    save_index(index, chunks, EMBEDDING_MODEL)

    print("\n✅ RAG index built successfully.")
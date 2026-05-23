"""
answer.py  –  RAG answer generation using Google Gemini (free tier).
Also contains the FastAPI /ask-complaints endpoint.
Run server:  uvicorn src.rag.answer:app --port 8001 --reload
"""

import os
import json
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
import sys
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rag.retrieve import load_index, load_embedding_model, retrieve, SIMILARITY_THRESHOLD

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.5-flash"   # free tier model
PROMPT_VERSION = "v1.0"

# ── Load RAG components at startup ────────────────────────────────────────────
index, chunks, index_meta = load_index()
embed_model = load_embedding_model(index_meta["embedding_model"])

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm = genai.GenerativeModel(MODEL_NAME)
else:
    llm = None
    print("⚠️  GEMINI_API_KEY not set — answers will be retrieval-only summaries")

# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:     str
    product:      Optional[str] = None
    company:      Optional[str] = None
    date_received: Optional[str] = None
    issue:        Optional[str] = None

class AskResponse(BaseModel):
    answer:               str
    evidence_ids:         list[str]
    evidence_sufficiency: str
    top_score:            float
    prompt_version:       str
    model_used:           str
    retrieval_count:      int
    latency_ms:           float

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Intelligence — RAG Service",
    description="Answers complaint intelligence questions with cited evidence.",
    version="1.0.0",
)

# ── Core answer function ──────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[dict]) -> str:
    context = ""
    for i, c in enumerate(chunks, 1):
        context += (
            f"\n[{i}] Complaint ID: {c['complaint_id']}\n"
            f"    Product: {c['product']} | Issue: {c['issue']}\n"
            f"    Company: {c['company']} | Date: {c['date_received']}\n"
            f"    Narrative excerpt: {c['text'][:300]}\n"
        )

    return f"""You are a financial complaint analyst. Answer the question below 
using ONLY the complaint records provided. Do not add information not found 
in the records. Be concise (3-5 sentences). Cite complaint IDs inline.

COMPLAINT RECORDS:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(question: str, filters: dict = None) -> dict:
    start = time.time()

    # Retrieve relevant chunks
    result = retrieve(question, index, chunks, embed_model, filters=filters)

    if not result["hit"]:
        return {
            "answer":               "I cannot answer this question as no sufficiently relevant complaint records were found. Please try a different question or broaden your filters.",
            "evidence_ids":         [],
            "evidence_sufficiency": "INSUFFICIENT — no chunks above similarity threshold",
            "top_score":            0.0,
            "prompt_version":       PROMPT_VERSION,
            "model_used":           "none — retrieval failed",
            "retrieval_count":      0,
            "latency_ms":           round((time.time() - start) * 1000, 2),
        }

    # Build sufficiency note
    top_score = result["top_score"]
    if top_score >= 0.6:
        sufficiency = f"SUFFICIENT — strong evidence (top score: {top_score})"
    elif top_score >= 0.4:
        sufficiency = f"MODERATE — partial evidence (top score: {top_score})"
    else:
        sufficiency = f"WEAK — low confidence evidence (top score: {top_score})"

    # Generate answer with LLM or fallback
    if llm:
        try:
            prompt   = build_prompt(question, result["retrieved"])
            response = llm.generate_content(prompt)
            answer   = response.text.strip()
            model_used = MODEL_NAME
        except Exception as e:
            answer     = f"LLM generation failed: {str(e)}. Retrieved {len(result['retrieved'])} relevant complaints."
            model_used = "fallback"
    else:
        # Fallback: summarize retrieved chunks without LLM
        ids     = ", ".join(result["evidence_ids"])
        answer  = (
            f"Based on {len(result['retrieved'])} retrieved complaint records "
            f"(IDs: {ids}), the complaints relate to: "
            + "; ".join(set(c["issue"] for c in result["retrieved"] if c.get("issue")))
            + ". Configure GEMINI_API_KEY for full answers."
        )
        model_used = "retrieval-only"

    return {
        "answer":               answer,
        "evidence_ids":         result["evidence_ids"],
        "evidence_sufficiency": sufficiency,
        "top_score":            top_score,
        "prompt_version":       PROMPT_VERSION,
        "model_used":           model_used,
        "retrieval_count":      len(result["retrieved"]),
        "latency_ms":           round((time.time() - start) * 1000, 2),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "index_version": index_meta.get("index_version", "1.0"),
        "total_chunks":  index_meta.get("total_chunks", 0),
        "model":         index_meta.get("embedding_model", ""),
    }


@app.post("/ask-complaints", response_model=AskResponse)
def ask_complaints(request: AskRequest):
    try:
        filters = {
            "product":       request.product,
            "company":       request.company,
            "date_received": request.date_received,
        }
        filters = {k: v for k, v in filters.items() if v}  # remove None
        result  = generate_answer(request.question, filters or None)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.rag.answer:app", host="0.0.0.0", port=8001, reload=True)
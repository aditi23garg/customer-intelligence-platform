"""
rag_eval.py  –  10 question evaluation for the RAG service.
Run:  python src/rag/rag_eval.py
"""

import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.rag.retrieve import load_index, load_embedding_model, retrieve
from src.rag.answer import generate_answer

ROOT     = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── 10 Eval Questions ─────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What are the most common complaints about credit reporting?",
        "expected_topics": ["credit report", "reporting", "investigation"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q02",
        "question": "What issues do customers face with mortgage payments?",
        "expected_topics": ["mortgage", "payment", "loan"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q03",
        "question": "How do companies typically respond to credit card complaints?",
        "expected_topics": ["credit card", "response", "closed"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q04",
        "question": "What problems do customers report about debt collection practices?",
        "expected_topics": ["debt", "collection", "contact"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q05",
        "question": "What are common issues with student loan servicing?",
        "expected_topics": ["student loan", "servicing", "payment"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q06",
        "question": "What bank account problems do customers complain about most?",
        "expected_topics": ["bank account", "checking", "deposit"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q07",
        "question": "How long does it typically take companies to resolve complaints?",
        "expected_topics": ["response", "closed", "timely"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q08",
        "question": "What complaints exist about payday loans or personal loans?",
        "expected_topics": ["loan", "interest", "payment"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q09",
        "question": "What issues do customers have with money transfers or remittances?",
        "expected_topics": ["transfer", "money", "transaction"],
        "min_evidence_ids": 1,
    },
    {
        "id": "Q10",
        "question": "What is the weather like on Mars?",   # adversarial — should get weak/no retrieval
        "expected_topics": [],
        "min_evidence_ids": 0,
        "expect_refusal": True,
    },
]


# ── Evaluator ─────────────────────────────────────────────────────────────────
def evaluate_question(q: dict, index, chunks, embed_model) -> dict:
    start  = time.time()
    result = generate_answer(q["question"])
    elapsed = round((time.time() - start) * 1000, 2)

    answer       = result["answer"].lower()
    evidence_ids = result["evidence_ids"]
    top_score    = result["top_score"]
    hit          = result["retrieval_count"] > 0

    # Check 1: refusal test
    if q.get("expect_refusal"):
        passed = not hit or top_score < 0.35
        notes  = "PASS — correctly refused/weak retrieval for off-topic question" \
                 if passed else "FAIL — should have refused but retrieved with high confidence"
        return {
            "id":           q["id"],
            "question":     q["question"],
            "passed":       passed,
            "notes":        notes,
            "evidence_ids": evidence_ids,
            "top_score":    top_score,
            "latency_ms":   elapsed,
        }

    # Check 2: did we get evidence?
    has_evidence = len(evidence_ids) >= q["min_evidence_ids"]

    # Check 3: does answer relate to expected topics?
    topic_hits = [t for t in q["expected_topics"] if t.lower() in answer]
    topic_ok   = len(topic_hits) > 0 if q["expected_topics"] else True

    passed = has_evidence and topic_ok and hit
    notes_parts = []
    if not hit:
        notes_parts.append("FAIL — no retrieval hit")
    if not has_evidence:
        notes_parts.append(f"FAIL — expected {q['min_evidence_ids']} evidence IDs, got {len(evidence_ids)}")
    if not topic_ok:
        notes_parts.append(f"FAIL — none of expected topics {q['expected_topics']} found in answer")
    if passed:
        notes_parts.append(f"PASS — retrieved {len(evidence_ids)} IDs, topics matched: {topic_hits}")

    return {
        "id":           q["id"],
        "question":     q["question"],
        "passed":       passed,
        "notes":        " | ".join(notes_parts),
        "evidence_ids": evidence_ids,
        "top_score":    top_score,
        "latency_ms":   elapsed,
        "answer":       result["answer"][:200] + "...",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("RAG EVALUATION — 10 Questions")
    print("="*60)

    index, chunks, meta = load_index()
    embed_model = load_embedding_model(meta["embedding_model"])

    results  = []
    passed   = 0

    for q in EVAL_QUESTIONS:
        print(f"\n  Running {q['id']}: {q['question'][:60]}...")
        r = evaluate_question(q, index, chunks, embed_model)
        results.append(r)
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon} {r['id']} — {r['notes']}")
        if r["passed"]:
            passed += 1

    total = len(EVAL_QUESTIONS)
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{total} passed")
    print("="*60)

    # Save report
    report = {
        "total":   total,
        "passed":  passed,
        "failed":  total - passed,
        "results": results,
    }
    out = DOCS_DIR / "rag_eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out}")
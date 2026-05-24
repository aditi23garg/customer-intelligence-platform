"""
rag_monitor.py  –  RAG monitoring metrics.
Tracks hit rate, latency, refusal rate, avg score, token count.
Run:  python monitoring/rag_monitor.py
"""

import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.rag.retrieve import load_index, load_embedding_model, retrieve
from src.rag.answer import generate_answer

REPORTS = ROOT / "docs"
REPORTS.mkdir(exist_ok=True)

# Test questions for monitoring
MONITOR_QUESTIONS = [
    "What are common credit reporting complaints?",
    "How do banks respond to mortgage complaints?",
    "What issues exist with debt collection?",
    "Tell me about credit card billing disputes.",
    "What complaints exist about loan modifications?",
    "How are student loan complaints handled?",
    "What is the weather on Jupiter?",        # should miss
    "Tell me a joke about finance.",           # should miss
]


def run_monitoring() -> dict:
    index, chunks, meta = load_index()
    embed_model = load_embedding_model(meta["embedding_model"])

    results     = []
    total       = len(MONITOR_QUESTIONS)
    hits        = 0
    refusals    = 0
    latencies   = []
    scores      = []

    print(f"  Running {total} monitoring questions...")

    for q in MONITOR_QUESTIONS:
        start  = time.time()
        result = generate_answer(q)
        elapsed = round((time.time() - start) * 1000, 2)

        hit      = result["retrieval_count"] > 0
        refused  = not hit or result["top_score"] < 0.30
        latencies.append(elapsed)

        if hit:
            hits += 1
            scores.append(result["top_score"])
        if refused:
            refusals += 1

        results.append({
            "question":        q,
            "hit":             hit,
            "refused":         refused,
            "top_score":       result["top_score"],
            "retrieval_count": result["retrieval_count"],
            "latency_ms":      elapsed,
        })

        icon = "✅" if hit and not refused else "⚠️"
        print(f"    {icon} [{elapsed:.0f}ms] score={result['top_score']:.3f} — {q[:50]}")

    # Summary metrics
    summary = {
        "total_questions":    total,
        "hit_count":          hits,
        "hit_rate":           round(hits / total, 4),
        "refusal_count":      refusals,
        "refusal_rate":       round(refusals / total, 4),
        "avg_latency_ms":     round(sum(latencies) / len(latencies), 2),
        "max_latency_ms":     round(max(latencies), 2),
        "avg_top_score":      round(sum(scores) / len(scores), 4) if scores else 0,
        "empty_retrievals":   total - hits,
        "results":            results,
    }

    return summary


if __name__ == "__main__":
    print("="*60)
    print("RAG MONITORING METRICS")
    print("="*60)

    print("\nRunning monitoring questions...")
    summary = run_monitoring()

    print("\n" + "="*60)
    print("RAG MONITORING SUMMARY")
    print("="*60)
    print(f"  Hit rate        : {summary['hit_rate']:.1%} ({summary['hit_count']}/{summary['total_questions']})")
    print(f"  Refusal rate    : {summary['refusal_rate']:.1%}")
    print(f"  Avg latency     : {summary['avg_latency_ms']:.0f}ms")
    print(f"  Max latency     : {summary['max_latency_ms']:.0f}ms")
    print(f"  Avg top score   : {summary['avg_top_score']:.4f}")
    print(f"  Empty retrievals: {summary['empty_retrievals']}")

    # Save report
    out = REPORTS / "rag_monitoring_report.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Report saved → {out}")
    print("\n✅ RAG monitoring complete.")
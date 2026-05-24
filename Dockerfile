# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn pandas scikit-learn xgboost joblib pydantic \
    python-multipart google-generativeai sentence-transformers \
    faiss-cpu python-dotenv

# ── Copy source code and artifacts ───────────────────────────────────────────
COPY src/ ./src/
COPY data/artifacts/ ./data/artifacts/
COPY data/models/ ./data/models/
COPY data/rag_index/ ./data/rag_index/
COPY data/samples/bank_marketing_sample.csv ./data/samples/
# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.serving.serve:app", "--host", "0.0.0.0", "--port", "8000"]
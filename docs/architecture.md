# Architecture — Customer Intelligence Platform

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER / ANALYST                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Browser
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND (Azure Container Apps)              │
│         customer-intel-frontend.wonderfulriver-676a1274...          │
│                                                                     │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │  Dashboard  │  │ ML Predictor │  │   Complaint Intel (RAG) │   │
│   └─────────────┘  └──────────────┘  └────────────────────────┘   │
│                     ┌──────────────┐                               │
│                     │Customer Intel│                               │
│                     └──────────────┘                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS REST
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               FASTAPI BACKEND (Azure Container Apps)                │
│            customer-intel-ml.wonderfulriver-676a1274...             │
│                                                                     │
│  GET  /health          POST /predict        POST /batch-score       │
│  POST /ask-complaints  POST /customer-intel                         │
│                                                                     │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐    │
│  │       ML LANE            │  │         RAG LANE             │    │
│  │                          │  │                              │    │
│  │  ┌────────────────────┐  │  │  ┌────────────────────────┐ │    │
│  │  │  XGBoost Classifier│  │  │  │  FAISS Index (flat IP) │ │    │
│  │  │  improved_model    │  │  │  │  ~18,000 chunks        │ │    │
│  │  │  .joblib           │  │  │  │  complaints.index      │ │    │
│  │  └────────────────────┘  │  │  └────────────────────────┘ │    │
│  │  ┌────────────────────┐  │  │  ┌────────────────────────┐ │    │
│  │  │  Feature Pipeline  │  │  │  │  sentence-transformers  │ │    │
│  │  │  features.py       │  │  │  │  all-MiniLM-L6-v2      │ │    │
│  │  │  encoders.json     │  │  │  └────────────────────────┘ │    │
│  │  └────────────────────┘  │  │  ┌────────────────────────┐ │    │
│  └──────────────────────────┘  │  │  Gemini 2.5 Flash (LLM)│ │    │
│                                │  │  grounded generation   │ │    │
│                                │  └────────────────────────┘ │    │
│                                └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                                  │
│                                                                     │
│  UCI Bank Marketing CSV  →  ingest.py  →  validate.py              │
│  CFPB Complaints CSV     →  ingest.py  →  validate.py              │
│                                    ↓                                │
│                              features.py                            │
│                          (reusable transforms)                      │
│                                    ↓                                │
│              ┌─────────────────────┴──────────────────┐            │
│              ▼                                         ▼            │
│          train.py                               build_index.py      │
│    baseline + improved model              chunk + embed + FAISS     │
│          MLflow tracking                    sentence-transformers   │
│              ↓                                         ↓            │
│        promotion gate                           retrieve.py         │
│     (PR-AUC +3%, F1 -2%)                      answer.py            │
│              ↓                                         ↓            │
│        model artifacts                          rag_eval.py         │
│    (joblib + metadata JSON)                  (10 Q eval + report)  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD + REGISTRY                                 │
│                                                                     │
│  GitHub Push → GitHub Actions CI                                    │
│                    ├── pytest tests/                                │
│                    └── python src/data_pipeline/validate.py        │
│                                                                     │
│  GitHub Push → Deploy Streamlit Frontend workflow                   │
│                    ├── docker build Dockerfile.streamlit            │
│                    ├── docker push → Azure Container Registry       │
│                    └── az containerapp update                       │
│                                                                     │
│  Azure Container Registry: customerintelacr2.azurecr.io            │
│    ├── customer-intel-ml:v4      (ML + RAG API)                    │
│    └── customer-intel-frontend:latest  (Streamlit UI)              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       MONITORING                                    │
│                                                                     │
│  ml_drift.py  →  Evidently report  →  docs/ml_drift_report.html   │
│                  (simulated shift on age, balance, campaign,        │
│                   duration — 4/7 features drifted → RETRAIN)       │
│                                                                     │
│  rag_monitor.py  →  docs/rag_monitoring_report.json                │
│                  (hit rate: 87.5%, avg latency: 3982ms,            │
│                   refusal rate: 12.5%, avg score: 0.6418)          │
│                                                                     │
│  Retrain trigger: manual on RETRAIN recommendation                 │
│  (automated trigger is stretch — not yet implemented)              │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow — /predict

```
POST /predict (customer features JSON)
        │
        ▼
  PredictRequest (Pydantic validation)
        │
        ▼
  build_features(df, encoders, fit=False)
  ├── clean_data()
  ├── encode_target()        ← dummy, dropped before predict
  ├── engineer_numeric_features()
  │     ├── fe_never_contacted
  │     ├── fe_contact_intensity
  │     ├── fe_balance_positive
  │     ├── fe_long_call
  │     └── fe_age_group
  └── encode_categoricals(fit=False)  ← uses saved encoders.json
        │
        ▼
  XGBoost.predict_proba(X)
        │
        ▼
  prob >= 0.5 → prediction (0 or 1)
        │
        ▼
  PredictResponse
  ├── prediction (0/1)
  ├── probability (float)
  ├── threshold (0.5)
  ├── decision (WILL/WILL NOT SUBSCRIBE)
  ├── model_name
  └── model_version
```

## Data Flow — /ask-complaints

```
POST /ask-complaints (question + optional filters)
        │
        ▼
  encode query → sentence-transformers embedding
        │
        ▼
  FAISS.search(query_embedding, top_k=50)
        │
        ▼
  Apply filters (product, company, date)
        │
        ▼
  Score threshold check (>= 0.30)
        │
   ┌────┴────┐
   │         │
  HIT      MISS
   │         │
   ▼         ▼
Gemini    REFUSE
generate  (no chunks
answer    above threshold)
   │
   ▼
  AskResponse
  ├── answer (grounded text)
  ├── evidence_ids (complaint IDs cited)
  ├── evidence_sufficiency (SUFFICIENT/MODERATE/WEAK)
  ├── top_score
  ├── prompt_version
  ├── model_used
  ├── retrieval_count
  └── latency_ms
```

## Deployment Architecture

```
Azure Subscription (Student — Southeast Asia)
│
├── Resource Group: customer-intel-rg2
│   │
│   ├── Azure Container Registry: customerintelacr2
│   │   ├── customer-intel-ml:v1, v2, v3, v4
│   │   └── customer-intel-frontend:latest
│   │
│   ├── Container Apps Environment: customer-intel-env
│   │   │
│   │   ├── Container App: customer-intel-ml
│   │   │   ├── Image: customer-intel-ml:v4
│   │   │   ├── CPU: 1.0, Memory: 2Gi
│   │   │   ├── Min replicas: 1, Max: 1
│   │   │   ├── Port: 8000
│   │   │   └── Env: GEMINI_API_KEY
│   │   │
│   │   └── Container App: customer-intel-frontend
│   │       ├── Image: customer-intel-frontend:latest
│   │       ├── Port: 8501
│   │       └── Env: API_BASE_URL
│   │
│   └── Log Analytics Workspace (auto-created)
│
└── GitHub Actions (CI/CD)
    ├── CI Pipeline — runs on every push
    └── Deploy Streamlit Frontend — runs on every push
```

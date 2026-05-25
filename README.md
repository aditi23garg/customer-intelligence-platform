# Customer Intelligence Platform
> **Futurense AI Clinic · IIT Gandhinagar · Week 13 Mini-Project**

A production-minded ML + LLM/RAG platform for Meridian Financial. Predicts campaign conversion using structured customer data and answers complaint intelligence questions over real CFPB narratives with cited evidence.

---

## Live Demo

| Service | URL |
|---|---|
| **Streamlit Frontend** | https://customer-intel-frontend.wonderfulriver-676a1274.southeastasia.azurecontainerapps.io |
| **ML + RAG API** | https://customer-intel-ml.wonderfulriver-676a1274.southeastasia.azurecontainerapps.io |
| **API Docs (Swagger)** | https://customer-intel-ml.wonderfulriver-676a1274.southeastasia.azurecontainerapps.io/docs |

---

## What It Does

**ML Lane** — Predicts whether a customer will subscribe to a term deposit (yes/no) using the UCI Bank Marketing dataset. Trains a baseline Logistic Regression and an improved XGBoost model, tracked with MLflow. A relative promotion gate ensures only the better model gets deployed.

**RAG Lane** — Answers complaint intelligence questions over 5,000 CFPB consumer complaint narratives. Uses FAISS vector search + sentence-transformers for retrieval and Gemini for grounded, cited answers. Refuses when no relevant evidence is found.

**Integration** — A unified `/customer-intel` endpoint combines ML conversion prediction and RAG complaint themes for a given customer + segment filter in one response.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│         (Azure Container Apps · Southeast Asia)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    REST API calls
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                             │
│         (Azure Container Apps · Southeast Asia)             │
│                                                             │
│   ┌─────────────────┐      ┌──────────────────────────┐    │
│   │    ML Service   │      │       RAG Service         │    │
│   │                 │      │                           │    │
│   │  XGBoost model  │      │  FAISS index (5k chunks)  │    │
│   │  POST /predict  │      │  sentence-transformers    │    │
│   │  POST /batch    │      │  Gemini 2.5 Flash         │    │
│   └────────┬────────┘      └──────────┬────────────────┘    │
│            │                          │                      │
│            └──────────┬───────────────┘                      │
│                       │                                      │
│              POST /customer-intel                            │
└───────────────────────┴──────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼──────┐             ┌────────▼────────┐
   │   MLflow    │             │   GitHub Actions │
   │  Tracking   │             │   CI/CD Pipeline │
   └─────────────┘             └─────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status, model name, version |
| POST | `/predict` | Predict campaign conversion for a customer |
| POST | `/batch-score` | Score a CSV/JSON batch of customers |
| POST | `/ask-complaints` | Answer a complaint question with cited evidence |
| POST | `/customer-intel` | Combined ML band + RAG complaint themes |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repo

```bash
git clone https://github.com/aditi23garg/customer-intelligence-platform.git
cd customer-intelligence-platform
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 5. Download data

```bash
python src/data_pipeline/ingest.py
```

> Place `bank-full.csv` and `complaints.csv` in `data/samples/` before running.
> Download links are in `data/README.md`.

### 6. Run validation

```bash
python src/data_pipeline/validate.py
```

### 7. Build features + train models

```bash
python src/data_pipeline/features.py
python src/training/train.py
```

### 8. Build RAG index

```bash
python src/rag/build_index.py
```

### 9. Start the API

```bash
uvicorn src.serving.serve:app --reload --port 8000
```

### 10. Start the frontend

```bash
streamlit run app/streamlit_app.py
```

---

## Run with Docker

```bash
docker build -t customer-intel .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key customer-intel
```

Or with Docker Compose:

```bash
docker-compose up
```

---

## Project Structure

```
customer-intelligence-platform/
├── app/
│   └── streamlit_app.py          # Streamlit frontend
├── data/
│   ├── samples/                  # Small data samples (committed)
│   ├── artifacts/                # Encoders, gate results, metadata
│   ├── models/                   # Trained model files
│   ├── rag_index/                # FAISS index + chunk store
│   └── README.md                 # Data sources and download instructions
├── src/
│   ├── data_pipeline/
│   │   ├── ingest.py             # Download + sample both datasets
│   │   ├── validate.py           # Schema + business rule validation
│   │   └── features.py           # Reusable feature engineering
│   ├── training/
│   │   ├── train.py              # Train baseline + improved model
│   │   └── evaluate.py           # Metrics, promotion gate
│   ├── serving/
│   │   ├── serve.py              # FastAPI app (all endpoints)
│   │   ├── serve_ml_only.py      # Lightweight ML-only version
│   │   └── schemas.py            # Pydantic request/response schemas
│   └── rag/
│       ├── build_index.py        # Chunk + embed + FAISS index
│       ├── retrieve.py           # Similarity search with filters
│       ├── answer.py             # Gemini grounded answer generation
│       └── rag_eval.py           # 10-question RAG evaluation
├── tests/
│   ├── test_features.py          # Feature engineering unit tests
│   ├── test_schemas.py           # API schema validation tests
│   └── test_retrieval.py        # RAG retrieval tests
├── monitoring/
│   ├── ml_drift.py               # Evidently drift report
│   └── rag_monitor.py            # RAG metrics monitoring
├── docs/
│   ├── architecture.md           # Architecture diagram
│   ├── decision_log.md           # Rejected approaches + rationale
│   ├── rag_eval_report.json      # RAG evaluation results
│   ├── ml_drift_report.html      # Evidently drift report
│   └── ml_drift_summary.json     # Drift summary
├── .github/
│   └── workflows/
│       ├── ci.yml                # CI: tests + validation on push
│       └── deploy-frontend.yml   # Deploy Streamlit to Azure
├── Dockerfile                    # ML + RAG API container
├── Dockerfile.streamlit          # Frontend container
├── docker-compose.yml            # Local orchestration
├── reflection.md                 # Project reflection
└── README.md
```

---

## Datasets

| Dataset | Use | Source |
|---|---|---|
| UCI Bank Marketing | ML model training | [UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| CFPB Consumer Complaints | RAG Q&A over complaint narratives | [CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/) |

Only small samples (5,000 rows each) are committed to Git. Full datasets must be downloaded separately — see `data/README.md`.

---

## Model Performance

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Baseline (Logistic Regression) | 0.8749 | 0.8511 | 0.8118 |
| **Improved (XGBoost)** | **0.9215** | **0.8912** | **0.8511** |

**Promotion gate:** Improved model promoted — PR-AUC improved by +0.04 (threshold: +0.03) and F1 improved by +0.04 (max drop allowed: -0.02).

---

## RAG Evaluation

| Metric | Value |
|---|---|
| Questions evaluated | 10 |
| Pass rate | 8/10 (80%) |
| Avg top similarity score | 0.6418 |
| Refusal on off-topic | ✅ Working |
| Evidence IDs cited | ✅ Every answer |

---

## Monitoring

**ML Drift** — Simulated production shift detected drift in 4/7 features (age, balance, campaign, duration). Recommendation: RETRAIN. Full HTML report in `docs/ml_drift_report.html`.

**RAG Monitoring** — Hit rate: 87.5%, avg latency: 3,982ms, refusal rate: 12.5%, avg top score: 0.6418.

---

## CI/CD

GitHub Actions runs on every push to `main`:
1. Install dependencies
2. Run unit tests (`pytest tests/`)
3. Run data validation (`python src/data_pipeline/validate.py`)

A separate workflow deploys the Streamlit frontend to Azure Container Apps on every push.

---

## Deployment

| Component | Platform | Image |
|---|---|---|
| ML + RAG API | Azure Container Apps (Southeast Asia) | `customerintelacr2.azurecr.io/customer-intel-ml:v4` |
| Streamlit Frontend | Azure Container Apps (Southeast Asia) | `customerintelacr2.azurecr.io/customer-intel-frontend:latest` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Training | scikit-learn, XGBoost, MLflow |
| Feature Engineering | pandas, numpy |
| Data Validation | pandera |
| Vector Search | FAISS (flat inner product) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Google Gemini 2.5 Flash |
| API | FastAPI, uvicorn, pydantic |
| Frontend | Streamlit |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Cloud | Azure Container Apps, Azure Container Registry |

---

## Environment Variables

```bash
GEMINI_API_KEY=your_gemini_api_key
```

Never commit `.env` — use `.env.example` as a template.

---

## Reflection

See `reflection.md` for answers to:
- Why XGBoost over alternatives
- What broke first during deployment
- Gate margin rationale
- RAG failure analysis
- Production risks
- What a senior MLOps engineer would criticize

---

## Author

**Aditi Garg** · IIT Gandhinagar · Futurense AI Clinic Week 13

# Decision Log — Customer Intelligence Platform
> Rejected approaches, rationale, known limitations, and hardening plan.

---

## Rejected Approaches

### 1. Neural Network for ML Lane
**Considered:** PyTorch MLP for campaign conversion prediction.
**Rejected because:** Overkill for 5,000 tabular rows. Neural networks need large datasets to outperform gradient boosting on tabular data. XGBoost achieves 0.92 ROC-AUC with far less tuning, faster inference, and built-in feature importance. Neural networks also add deployment complexity (model serialization, ONNX conversion) with no measurable benefit.

### 2. OpenAI GPT-4 for RAG generation
**Considered:** Using OpenAI's API instead of Google Gemini.
**Rejected because:** Cost. GPT-4 at ~$0.03/1K output tokens would exhaust the free tier quickly during development. Gemini 2.5 Flash is free tier with generous quotas and performs comparably for grounded Q&A over structured complaint text.

### 3. ChromaDB instead of FAISS
**Considered:** Using ChromaDB as the vector store (persistent, has metadata filtering built in).
**Rejected because:** ChromaDB requires a separate process or embedded mode with SQLite, adding deployment complexity. FAISS is a single file (`complaints.index`) that loads directly into memory — simpler to Docker-ize and copy into the container. Manual metadata filtering on the chunks list is sufficient for our filter requirements (product, company, date).

### 4. Azure AI Search for RAG
**Considered:** Using Azure Cognitive Search as the vector store (enterprise option).
**Rejected because:** Azure AI Search is expensive even on the free tier (limited to 50MB index). It also requires network calls for every retrieval, adding ~200-400ms latency vs. in-process FAISS which retrieves in <10ms. For a demo platform with 5,000 records, FAISS is the right choice.

### 5. Azure ML Managed Online Endpoints
**Considered:** Deploying via Azure ML SDK v2 with Managed Online Endpoints.
**Rejected because:** Azure student subscriptions have quota restrictions on ML compute. Azure Container Apps with a custom Docker image achieves the same result (live HTTPS endpoint) with fewer quota requirements and simpler configuration.

### 6. Separate ML and RAG containers
**Considered:** Deploying ML API and RAG API as two separate Container Apps.
**Rejected because:** Two containers means two cold-start delays, two sets of Azure credits consumed, and a more complex `/customer-intel` endpoint that would need to make inter-service calls. Combined into one container for simplicity. Documented as a known limitation (cannot scale independently).

### 7. LangChain for RAG pipeline
**Considered:** Using LangChain to build the retrieval-augmented generation pipeline.
**Rejected because:** LangChain adds significant abstraction overhead and a heavy dependency tree. Building retrieval, prompt construction, generation, and evaluation as separate explicit stages (`retrieve.py`, `answer.py`, `rag_eval.py`) makes each stage debuggable independently — which is what the project spec requires.

### 8. Streamlit Cloud for frontend deployment
**Considered:** Deploying Streamlit to Streamlit Community Cloud (free).
**Rejected because:** Streamlit Cloud has cold start issues and requires a public GitHub repo with secrets management limitations. Azure Container Apps keeps the entire stack on one cloud provider, simplifying monitoring and access control. Also demonstrates cloud deployment depth.

### 9. Full 1M row CFPB dataset
**Considered:** Ingesting the complete CFPB complaints database (~1M records).
**Rejected because:** The project spec explicitly recommends 5,000-25,000 records locally. Loading 1M rows would require ~8GB RAM for embedding generation and a FAISS index that would be too large to copy into a Docker container. 5,000 records with real narratives provides sufficient coverage for the demo.

### 10. SHAP explanations in /predict response
**Considered:** Including SHAP feature importance values in every `/predict` response.
**Rejected because:** SHAP computation adds ~500-800ms per prediction for XGBoost. Including it in every response would make the API too slow for production use. Documented as a stretch goal — SHAP should run as a batch job and store explanations for auditing, not inline in the prediction response.

---

## Known Limitations

### ML Lane
- **Small training set:** 5,000 stratified rows from a 45,000 row dataset. The model's generalization to the full distribution is not validated.
- **Threshold is fixed at 0.5:** For the real imbalanced dataset (~11% positive), a lower threshold (0.3-0.35) would likely improve business performance.
- **No automated retraining:** Drift is detected by code but retraining must be triggered manually. An automated trigger on drift threshold breach is planned but not implemented.
- **Duration leakage:** The `duration` feature (call duration in seconds) is known to be a data leakage risk — you only know the duration after the call, which means it cannot be used in pre-call targeting. It was retained for the demo because it significantly improves metrics, but in a real system it would be excluded from pre-call models.

### RAG Lane
- **No PII filtering:** CFPB narratives contain real consumer complaint text. No PII detection or redaction layer exists beyond the CFPB's own `XXXX` placeholders.
- **Static index:** The FAISS index is built once and baked into the Docker image. New complaints require a full index rebuild and redeployment.
- **Gemini rate limits:** The free tier Gemini API has rate limits. Under high load, generation requests will fail and fall back to retrieval-only summaries.
- **Evaluation keyword matching:** The 10-question RAG eval uses keyword matching for topic validation, which misses semantically correct answers that use different vocabulary (as seen in Q07 and Q09 failures).
- **No conversation memory:** Each `/ask-complaints` call is stateless. Follow-up questions don't have context from previous answers.

### System
- **Single replica:** Both Container Apps run with max 1 replica. Under concurrent load, requests will queue.
- **No authentication:** The API endpoints are publicly accessible with no API key or authentication. In production, all endpoints would require OAuth2 or API key authentication.
- **No request logging:** API requests are not logged to a persistent store. Monitoring relies on Azure Container Apps built-in metrics only.
- **Hardcoded configuration:** Thresholds, margins, and model parameters are hardcoded in source files rather than a config file.

---

## Hardening Plan

### Short term (next sprint)
1. Move model artifacts and FAISS index to Azure Blob Storage — decouple model updates from container rebuilds
2. Add API key authentication to all endpoints
3. Move all thresholds and parameters to `config.yaml`
4. Add automated drift detection on a weekly schedule with Slack alerting
5. Add PII detection layer using spaCy NER before indexing complaint narratives

### Medium term
1. Separate ML and RAG into independent services with independent scaling
2. Implement champion-challenger: shadow-score the new model against the current champion before promotion
3. Expand RAG eval to 20+ questions with LLM-as-judge scoring
4. Add SHAP explanations as a batch job — store per-prediction for audit
5. Implement conversation memory in the RAG service for multi-turn Q&A

### Long term
1. Automated retraining pipeline triggered by drift alert
2. A/B testing framework for model variants
3. Full observability stack: structured logging → Azure Monitor → dashboards
4. Segment error analysis — identify which customer segments the model performs worst on
5. Data versioning with DVC or Delta Lake for full reproducibility

---

## What We Would Do Differently

1. **Start with Docker from day one.** Building the Docker image after the code was already written revealed several import path issues and missing dependencies. Starting with a container-first approach would have caught these earlier.

2. **Use a proper secrets manager from the start.** Gemini API key, Azure credentials, and ACR passwords were managed ad-hoc. A proper setup with Azure Key Vault referenced from Container Apps environment variables would be more secure and easier to rotate.

3. **Version the FAISS index separately from the model.** The index and model are currently both in the Docker image. They have different update cadences — the model updates when data drifts, the index updates when new complaints arrive. They should be stored and versioned independently.

4. **Use `uv` instead of `pip` for faster dependency management.** The Docker build takes 5+ minutes largely due to pip resolving and installing packages. `uv` would cut this to under 1 minute.

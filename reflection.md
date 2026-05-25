# Reflection — Customer Intelligence Platform
> Week 13 Mini-Project · Futurense AI Clinic · IIT Gandhinagar

---

## 1. Why this model family and this threshold over the alternatives you weighed?

**Model family choice — XGBoost over alternatives:**

I evaluated three model families for the campaign conversion prediction task:

- **Logistic Regression (baseline)** — Fast, interpretable, and a good calibration baseline. It achieved ROC-AUC of 0.87 but struggled with the non-linear interactions between features like `duration`, `pdays`, and `balance`. It also required significant feature scaling and still failed to converge fully within 1,000 iterations, suggesting the feature space was too complex for a linear model without heavy engineering.

- **XGBoost (chosen)** — Gradient boosted trees handle mixed feature types, non-linear interactions, and class imbalance naturally. It achieved ROC-AUC of 0.92 and PR-AUC of 0.89 without requiring feature scaling. The improvement over the baseline was +0.04 PR-AUC and +0.04 F1 — both clearing the promotion gate margins. XGBoost also provides feature importance natively, which is valuable for business interpretability.

- **Random Forest (considered, not built)** — Would have been a reasonable alternative to XGBoost but tends to be slower at inference and less tunable. Given the time constraint, I focused on one strong improved model rather than two mediocre ones.

**Threshold choice — 0.5:**

The default threshold of 0.5 was chosen because the dataset was balanced (50/50 after stratified sampling). In a real imbalanced scenario (which the full UCI dataset is — only ~11% positive), I would lower the threshold to ~0.3-0.35 to increase recall, since missing a true subscriber (false negative) is more costly than incorrectly targeting a non-subscriber (false positive) in a marketing campaign context. This is documented as a known limitation.

---

## 2. What broke first when you tried to deploy, and what did you change?

**Several things broke in sequence:**

**Break 1 — SSL certificate error on UCI dataset download.**
The UCI server's SSL certificate had expired. Fixed by downloading the dataset manually and updating `ingest.py` to read from a local file instead of fetching from the URL.

**Break 2 — CFPB API returned empty JSON.**
The CFPB public API returned an empty response intermittently. Fixed by downloading the full `complaints.csv` directly and reading it in chunks (10,000 rows at a time) to avoid memory errors on the 700MB file.

**Break 3 — Git push rejected due to large files.**
The `complaints.csv` (223MB) and `bank-full.csv` (4.6MB) were accidentally committed to Git history. GitHub rejected the push. Fixed by using `git filter-branch` to rewrite history and remove the large files, then force-pushing.

**Break 4 — MLflow tracking URI failed on Windows.**
MLflow's `set_tracking_uri` with a Windows path (`C:\...`) was treated as a URI scheme starting with `C:` which MLflow didn't recognize. Fixed by prefixing with `file:///` and replacing backslashes with forward slashes.

**Break 5 — Azure Container Registry region blocked.**
The `centralindia` region was blocked by the Azure student subscription policy. Fixed by switching to `southeastasia` which was available.

**Break 6 — Docker image 3GB push timeout.**
Including `sentence-transformers` made the Docker image ~3.2GB. The initial push kept timing out. Fixed by re-authenticating with `az acr login` and pushing again. The FAISS index also needed to be explicitly copied into the container — it was missing from the first deployment, causing a `FileIOReader` crash.

**Break 7 — Streamlit Quick Questions not working.**
The quick question buttons set session state but Streamlit's rerun cleared the form input before the API call happened. Fixed by using `st.session_state["direct_question"]` which gets processed in the same render cycle as the API call, bypassing the form entirely.

---

## 3. Why your gate margin, and what fails if you tighten PR-AUC by another 2 points?

**Gate margins chosen:**
- PR-AUC improvement ≥ +0.03 (3 percentage points)
- F1 drop ≤ -0.02 (max 2 percentage point drop allowed)

**Rationale:**
The 3% PR-AUC margin was chosen because it represents a meaningful business improvement — not just statistical noise. A model that improves PR-AUC by only 1-2% on a 5,000 sample dataset may not generalize to production. The 2% F1 tolerance was added to allow the improved model some flexibility in the precision-recall tradeoff — XGBoost might sacrifice a small amount of F1 while gaining significantly on PR-AUC through better calibration.

**What fails if we tighten to +5% PR-AUC:**
The current XGBoost improved PR-AUC by +0.0401 (4.01%). If we tightened the gate to +5%, this run would have been **blocked** — the baseline would have been promoted instead. To pass a tighter gate we would need:
- More hyperparameter tuning (learning rate, depth, subsample)
- SMOTE or other imbalance handling on the full dataset
- Additional feature engineering (interaction terms, ratio features)
- A larger training set (currently only 4,000 rows after stratified sampling)

This shows the gate is appropriately calibrated — tight enough to block marginal improvements but not so tight it's impossible to pass without major effort.

---

## 4. Show one complaint answer your RAG got wrong or ungrounded. How did your eval or refusal rule catch it, or why did it slip through?

**Failed question — Q07: "How long does it typically take companies to resolve complaints?"**

**Expected topics:** response, closed, timely

**What happened:** The RAG retrieved 5 relevant complaint records with a top similarity score of 0.58. However the Gemini answer focused on the *types* of resolutions (monetary relief, non-monetary relief, closed with explanation) rather than the *time* it takes. The answer was technically grounded in the retrieved records — it cited real complaint IDs — but it answered a slightly different question than what was asked.

**Why it slipped through the eval:**
The eval check looked for keywords like "response", "closed", and "timely" in the answer text. The Gemini response used words like "resolved", "relief", and "addressed" instead — semantically similar but lexically different. The topic-matching check was too rigid (exact string match) rather than semantic.

**How to fix this:**
Use embedding similarity between expected topics and the answer text rather than exact keyword matching. Alternatively, use an LLM-as-judge approach where a second LLM call evaluates whether the answer addressed the question intent. This is logged in the decision log as a known limitation of the current eval harness.

**Refusal behavior:**
The refusal logic correctly triggered for Q10 ("What is the weather like on Mars?") — top score was 0.00, no chunks crossed the similarity threshold, and the answer correctly stated it could not find relevant complaint records. This demonstrates the refusal gate working as intended.

---

## 5. If this went live to real customers tomorrow, name the one risk you did not fully close.

**The one risk I did not fully close: PII exposure in RAG responses.**

The CFPB complaint narratives contain real consumer complaint text. While CFPB redacts obvious personal information with `XXXX` placeholders (account numbers, dates, names), the narratives still contain:

- Company-specific details that could identify individuals in small markets
- Financial amounts that, combined with other data, could be re-identifying
- Descriptions of personal circumstances (medical conditions mentioned in context of financial hardship)

The current system returns raw narrative excerpts as part of the RAG context and sometimes includes them verbatim in Gemini's answers. There is no PII-detection layer, no output filtering, and no audit trail of what text was exposed in responses.

**What would be needed before going live:**
1. A PII detection pass (using spaCy NER or a dedicated PII model) on all chunks before indexing
2. Output filtering on Gemini responses to catch any PII that slips through
3. Role-based access control — complaint details should only be accessible to authorized support staff, not all users
4. Response logging with PII masking for audit purposes
5. A legal review of whether using CFPB narratives in a commercial RAG system complies with the CFPB's data use terms

---

## 6. What would a senior MLOps engineer criticize first in your repo?

**Criticism 1 — No real-time monitoring or alerting.**
The drift report is generated by running a script manually against a simulated shift. In production, drift detection needs to run automatically on a schedule (daily or weekly) against real incoming data, with alerts sent to Slack or PagerDuty when drift exceeds the threshold. The current setup requires someone to remember to run `python monitoring/ml_drift.py`.

**Criticism 2 — Model artifacts are committed to the container image.**
The trained model files (`improved_model.joblib`, `baseline_model.joblib`) and the FAISS index are baked into the Docker image. This means every model update requires rebuilding and redeploying the entire container. In production, artifacts should be stored in Azure Blob Storage or MLflow Model Registry and loaded at startup — decoupling model updates from container rebuilds.

**Criticism 3 — No rollback mechanism.**
If the deployed model starts performing poorly (e.g., sudden spike in refusal rate or prediction distribution shift), there is no automated rollback to the previous model version. A champion-challenger setup with shadow scoring would catch regressions before they affect all users.

**Criticism 4 — Single container for ML + RAG.**
Running ML inference and RAG (which loads a 3GB sentence-transformers model) in the same container means they cannot be scaled independently. If complaint Q&A traffic spikes, the entire container scales — including the ML predictor which may not need more instances. These should be separate services.

**Criticism 5 — Hardcoded thresholds.**
The similarity threshold (0.30), promotion gate margins (0.03, 0.02), and confidence bands (HIGH ≥ 0.7, MEDIUM ≥ 0.4) are hardcoded in the source files. These should be in a config file (`config.yaml`) or environment variables so they can be tuned without code changes.

**What I would prioritize fixing first:**
Artifact storage in Azure Blob + MLflow Model Registry. Everything else builds on having a proper model versioning and retrieval system. Without it, every deployment is manual and error-prone.

---

## Summary

This project taught me that the distance between "trained a model" and "runs reliably in production" is enormous. The model training itself took less than 20% of the total effort. The remaining 80% was data pipeline reliability, Docker image optimization, Azure region restrictions, Git history cleanup, Streamlit session state bugs, and RAG evaluation design. That ratio — 20% model, 80% system — is exactly what the project brief warned about, and experiencing it firsthand was the most valuable outcome.

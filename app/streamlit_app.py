"""
streamlit_app.py  -  Customer Intelligence Platform
Clean, subtle dark UI
"""

import streamlit as st
import requests

API_BASE = "https://customer-intel-ml.wonderfulriver-676a1274.southeastasia.azurecontainerapps.io"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stHeader"] { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
.main .block-container { padding-top: 2rem !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

.stApp { background-color: #111827; }

[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Page heading style */
.page-heading {
    font-size: 26px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
    letter-spacing: -0.3px;
}
.page-subheading {
    font-size: 13px;
    color: rgba(255,255,255,0.35);
    margin-bottom: 28px;
}

/* Section heading */
.section-heading {
    font-size: 16px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 14px;
    margin-top: 4px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.35);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
}
.metric-delta { font-size: 12px; margin-top: 4px; }
.delta-up { color: #86efac; }
.delta-warn { color: #fcd34d; }

/* KV rows */
.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 13px;
}
.kv-row:last-child { border-bottom: none; }
.kv-key { color: rgba(255,255,255,0.4); }
.kv-val { color: #e2e8f0; font-weight: 500; font-family: monospace; font-size: 12px; }

/* Drift boxes */
.drift-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.drift-label {
    font-size: 10px;
    font-weight: 600;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.drift-value { font-size: 14px; font-weight: 600; color: #fcd34d; }
.drift-change { font-size: 11px; color: rgba(255,255,255,0.35); margin-top: 4px; }

.alert-warn {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-left: 3px solid #fcd34d;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #fcd34d;
    margin-top: 16px;
}

/* Result boxes */
.result-high {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-bottom: 14px;
}
.result-medium {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-bottom: 14px;
}
.result-low {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-bottom: 14px;
}
.result-title { font-size: 24px; font-weight: 700; color: #f1f5f9; margin-bottom: 4px; }
.result-sub { font-size: 13px; color: rgba(255,255,255,0.4); }

/* Status */
.status-online {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #86efac;
    display: inline-block;
}
.status-offline {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #fca5a5;
    display: inline-block;
}

/* Chat */
.chat-user {
    background: #1e293b;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px 10px 4px 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #c7d2fe;
    line-height: 1.5;
}
.chat-bot {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px 10px 10px 4px;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #e2e8f0;
    line-height: 1.7;
}
.chat-meta {
    font-size: 11px;
    color: rgba(255,255,255,0.3);
    margin-bottom: 8px;
}

/* Evidence pills */
.evidence-pill {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #a5b4fc;
    margin: 2px;
    font-family: monospace;
}
.sufficiency-note {
    font-size: 11px;
    color: rgba(255,255,255,0.3);
    border-left: 2px solid rgba(99,102,241,0.3);
    padding-left: 8px;
    margin-top: 10px;
    font-style: italic;
}

/* Quick question buttons */
.stButton > button.quick-q {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}

/* All buttons - subtle style */
.stButton > button {
    background: #1e293b !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2d3748 !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.18) !important;
}

/* Primary action button */
.primary-btn > div > button,
.stButton > button[kind="primary"] {
    background: #3730a3 !important;
    color: #e0e7ff !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
.primary-btn > div > button:hover {
    background: #312e81 !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}
.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Labels */
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stRadio label {
    color: rgba(255,255,255,0.4) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

h1, h2, h3 { color: #f1f5f9 !important; }

.stProgress > div > div > div {
    background: #4f46e5 !important;
    border-radius: 4px !important;
}

.custom-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 16px 0;
}

.theme-pill {
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 6px;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=15)
        return r.json()
    except:
        return None

def api_predict(payload):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_ask(payload):
    try:
        r = requests.post(f"{API_BASE}/ask-complaints", json=payload, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_customer_intel(payload):
    try:
        r = requests.post(f"{API_BASE}/customer-intel", json=payload, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">🏦 Customer Intel</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:rgba(255,255,255,0.3);margin-bottom:16px;">Meridian Financial</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🤖 ML Predictor", "💬 Complaint Intel", "🔍 Customer Intel"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    health = api_health()
    if health and health.get("status") == "ok":
        st.markdown('<div class="status-online">● API Online</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:6px;font-size:11px;color:rgba(255,255,255,0.3);">{health.get("model_name","")} · v{health.get("model_version","")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">● API Offline</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:rgba(255,255,255,0.25);line-height:2.2;">🧠 XGBoost · FAISS · Gemini<br>☁️ Azure Container Apps<br>🔄 GitHub Actions CI/CD</div>', unsafe_allow_html=True)


# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<div class="page-heading">Platform Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Customer Intelligence Platform · Live on Azure</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, delta, cls in [
        (c1, "ROC-AUC",     "0.92",   "↑ +0.05 vs baseline", "delta-up"),
        (c2, "PR-AUC",      "0.89",   "↑ +0.04 vs baseline", "delta-up"),
        (c3, "F1 Score",    "0.851",  "↑ +0.039 vs baseline", "delta-up"),
        (c4, "RAG Hit Rate","87.5%",  "7 / 8 eval questions", "delta-up"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-delta {cls}">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-heading">🤖 ML Model Performance</div>', unsafe_allow_html=True)
        for k, v in [("Algorithm","XGBoost Classifier"),("Training samples","4,000"),("Test samples","1,000"),("Threshold","0.5"),("Promotion gate","✅ PASSED"),("MLflow tracking","✅ Active"),("Drift status","⚠️ RETRAIN recommended")]:
            st.markdown(f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-heading">💬 RAG Service Status</div>', unsafe_allow_html=True)
        for k, v in [("Embedding model","all-MiniLM-L6-v2"),("Vector store","FAISS (flat IP)"),("Complaint records","5,000"),("Total chunks","~18,000"),("LLM","Gemini 2.5 Flash"),("Similarity threshold","0.30"),("Avg top score","0.6418")]:
            st.markdown(f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📉 Drift Detection Report</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:rgba(255,255,255,0.3);font-size:12px;margin-bottom:14px;">Simulated production shift · Evidently AI · 4/7 features drifted</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    for col, feat, change in [(d1,"AGE","41.2 → 48.7"),(d2,"BALANCE","1,548 → 431"),(d3,"CAMPAIGN","2.5 → 4.5"),(d4,"DURATION","384s → 268s")]:
        with col:
            st.markdown(f"""<div class="drift-box">
                <div class="drift-label">{feat}</div>
                <div class="drift-value">DRIFTED</div>
                <div class="drift-change">{change}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="alert-warn">⚠️ &nbsp;<strong>Recommendation: RETRAIN</strong> — 57.14% of monitored features have drifted beyond threshold.</div>', unsafe_allow_html=True)


# ── ML Predictor ──────────────────────────────────────────────────────────────
elif page == "🤖 ML Predictor":
    st.markdown('<div class="page-heading">ML Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Predict campaign conversion · XGBoost v1.0 · POST /predict</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown('<div class="section-heading">Customer Features</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            age      = st.number_input("Age", min_value=18, max_value=95, value=35)
            balance  = st.number_input("Balance (€)", value=1500)
            duration = st.number_input("Call Duration (sec)", min_value=0, value=200)
            campaign = st.number_input("Campaign Contacts", min_value=1, value=2)
            pdays    = st.number_input("Pdays", value=-1)
            previous = st.number_input("Previous Contacts", min_value=0, value=0)
            day      = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        with r2:
            job       = st.selectbox("Job", ["management","technician","blue-collar","admin.","services","retired","entrepreneur","self-employed","student","housemaid","unemployed","unknown"])
            marital   = st.selectbox("Marital Status", ["married","single","divorced"])
            education = st.selectbox("Education", ["tertiary","secondary","primary","unknown"])
            housing   = st.selectbox("Housing Loan", ["yes","no"])
            loan      = st.selectbox("Personal Loan", ["no","yes"])
            contact   = st.selectbox("Contact Type", ["cellular","telephone","unknown"])
            month     = st.selectbox("Month", ["may","jun","jul","aug","oct","nov","dec","jan","feb","mar","apr","sep"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction →", use_container_width=True)

        if predict_btn:
            payload = {
                "age": int(age), "job": job, "marital": marital,
                "education": education, "default": "no",
                "balance": int(balance), "housing": housing,
                "loan": loan, "contact": contact, "day": int(day),
                "month": month, "duration": int(duration),
                "campaign": int(campaign), "pdays": int(pdays),
                "previous": int(previous), "poutcome": "unknown"
            }
            with st.spinner("Analyzing customer profile..."):
                result = api_predict(payload)
            st.session_state["predict_result"] = result

    with col2:
        result = st.session_state.get("predict_result")

        if result is None:
            st.markdown("""<div style="text-align:center;padding:60px 20px;background:#1e293b;
                border:1px solid rgba(255,255,255,0.06);border-radius:12px;margin-top:28px;">
                <div style="font-size:36px;margin-bottom:12px; opacity:0.4;">🎯</div>
                <div style="color:rgba(255,255,255,0.25);font-size:13px;">
                    Fill in customer features and click Run Prediction
                </div>
            </div>""", unsafe_allow_html=True)
        elif "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            prob = result.get("probability", 0)
            pred = result.get("prediction", 0)
            pct  = round(prob * 100)

            if pred == 1:
                css, icon, label, color = "result-high", "✅", "WILL SUBSCRIBE", "#86efac"
            else:
                css, icon, label, color = "result-low", "❌", "WILL NOT SUBSCRIBE", "#fca5a5"

            st.markdown(f"""<div class="{css}">
                <div style="font-size:32px;margin-bottom:10px;">{icon}</div>
                <div class="result-title">{label}</div>
                <div class="result-sub">Campaign conversion prediction</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:13px;margin-bottom:6px;">Probability: <strong style="color:{color};">{pct}%</strong></div>', unsafe_allow_html=True)
            st.progress(prob)
            st.markdown("<br>", unsafe_allow_html=True)
            for k, v in [("Probability", f"{prob:.4f}"), ("Threshold", result.get("threshold", 0.5)), ("Model", result.get("model_name","improved_xgboost")), ("Version", f"v{result.get('model_version','1.0')}")]:
                st.markdown(f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>', unsafe_allow_html=True)


# ── Complaint Intel ───────────────────────────────────────────────────────────
elif page == "💬 Complaint Intel":
    st.markdown('<div class="page-heading">Complaint Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Grounded Q&A over CFPB complaint narratives · FAISS + Gemini · POST /ask-complaints</div>', unsafe_allow_html=True)

    # Filters row
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc2:
        filter_product = st.selectbox("Product Filter", ["All Products","Credit card","Mortgage","Student loan","Credit reporting","Debt collection","Bank account","Personal loan"], key="fp")
    with fc3:
        filter_company = st.selectbox("Company Filter", ["All Companies","TRANSUNION","EQUIFAX","EXPERIAN","BANK OF AMERICA","WELLS FARGO","JPMORGAN CHASE","CITIBANK","CAPITAL ONE"], key="fc")

    # Question heading + input
    st.markdown('<div class="section-heading">Ask a Question</div>', unsafe_allow_html=True)

    with st.form("complaint_form", clear_on_submit=True):
        question = st.text_input(
            "question_input",
            placeholder="e.g. What are the most common complaints about credit reporting?",
            label_visibility="collapsed"
        )
        col_send, col_clear, col_space = st.columns([1, 1, 4])
        with col_send:
            submitted = st.form_submit_button("Send →", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("Clear", use_container_width=True)

    if clear:
        st.session_state["chat_history"] = []
        st.rerun()

    if submitted and question.strip():
        st.session_state["direct_question"] = question

    # Quick questions — 2 rows of 2 so full text is visible
    st.markdown('<div style="font-size:11px;color:rgba(255,255,255,0.3);font-weight:500;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px;margin-top:4px;">Quick Questions</div>', unsafe_allow_html=True)

    quick_qs = [
        "What are the most common complaints about credit reporting?",
        "How do companies typically respond to mortgage complaints?",
        "What problems do customers report about debt collection practices?",
        "What are common issues with student loan servicing?",
    ]

    qr1c1, qr1c2 = st.columns(2)
    qr2c1, qr2c2 = st.columns(2)
    q_cols = [qr1c1, qr1c2, qr2c1, qr2c2]

    for i, (col, q) in enumerate(zip(q_cols, quick_qs)):
        with col:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                st.session_state["direct_question"] = q

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Process question
    if "direct_question" in st.session_state:
        q    = st.session_state.pop("direct_question")
        prod = st.session_state.get("fp", "All Products")
        comp = st.session_state.get("fc", "All Companies")

        payload = {"question": q}
        if prod and prod != "All Products":  payload["product"] = prod
        if comp and comp != "All Companies": payload["company"] = comp

        with st.spinner(f"Searching {5000:,} complaint records..."):
            result = api_ask(payload)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        st.session_state["chat_history"].insert(0, {"question": q, "result": result})

    # Chat history
    history = st.session_state.get("chat_history", [])

    if not history:
        st.markdown("""<div style="text-align:center;padding:48px;background:#1e293b;
            border:1px solid rgba(255,255,255,0.05);border-radius:12px;">
            <div style="font-size:32px;margin-bottom:10px;opacity:0.3;">💬</div>
            <div style="color:rgba(255,255,255,0.25);font-size:13px;">
                Type a question or pick one from the quick questions above
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for item in history:
            st.markdown(f'<div class="chat-user">🧑 {item["question"]}</div>', unsafe_allow_html=True)
            r = item["result"]
            if "error" in r:
                st.error(r["error"])
            else:
                answer       = r.get("answer", "No answer.")
                evidence_ids = r.get("evidence_ids", [])
                sufficiency  = r.get("evidence_sufficiency", "")
                latency      = r.get("latency_ms", 0)
                model_used   = r.get("model_used", "gemini")
                retrieval    = r.get("retrieval_count", 0)
                pills        = "".join([f'<span class="evidence-pill">#{i}</span>' for i in evidence_ids])

                st.markdown(f"""<div class="chat-bot">
                    <div class="chat-meta">🤖 {model_used} &nbsp;·&nbsp; {round(latency)}ms &nbsp;·&nbsp; {retrieval} records retrieved</div>
                    {answer}
                    <div style="margin-top:10px;">{pills}</div>
                    <div class="sufficiency-note">{sufficiency}</div>
                </div>""", unsafe_allow_html=True)


# ── Customer Intel ────────────────────────────────────────────────────────────
elif page == "🔍 Customer Intel":
    st.markdown('<div class="page-heading">Customer Intel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheading">Combined ML prediction + RAG complaint themes · POST /customer-intel</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-heading">👤 Customer Profile</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            ci_age      = st.number_input("Age", min_value=18, max_value=95, value=42, key="ci_age")
            ci_balance  = st.number_input("Balance (€)", value=2000, key="ci_bal")
            ci_duration = st.number_input("Duration (sec)", min_value=0, value=250, key="ci_dur")
            ci_campaign = st.number_input("Campaign", min_value=1, value=2, key="ci_camp")
        with r2:
            ci_job     = st.selectbox("Job", ["management","technician","blue-collar","admin.","services","retired","entrepreneur","self-employed"], key="ci_job")
            ci_edu     = st.selectbox("Education", ["tertiary","secondary","primary","unknown"], key="ci_edu")
            ci_marital = st.selectbox("Marital", ["married","single","divorced"], key="ci_mar")
            ci_housing = st.selectbox("Housing Loan", ["yes","no"], key="ci_hou")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🔍 Complaint Segment Filter</div>', unsafe_allow_html=True)
        ci_product = st.selectbox("Product", ["All Products","Credit card","Mortgage","Student loan","Credit reporting","Debt collection","Bank account"], key="ci_prod")
        ci_issue   = st.text_input("Issue (optional)", placeholder="e.g. Billing dispute", key="ci_issue")

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Customer →", use_container_width=True)

        if analyze_btn:
            payload = {
                "age": int(ci_age), "job": ci_job, "marital": ci_marital,
                "education": ci_edu, "default": "no",
                "balance": int(ci_balance), "housing": ci_housing,
                "loan": "no", "contact": "cellular", "day": 15,
                "month": "may", "duration": int(ci_duration),
                "campaign": int(ci_campaign), "pdays": -1,
                "previous": 0, "poutcome": "unknown"
            }
            if ci_product and ci_product != "All Products": payload["product"] = ci_product
            if ci_issue: payload["issue"] = ci_issue

            with st.spinner("Running ML + RAG analysis..."):
                result = api_customer_intel(payload)
            st.session_state["ci_result"] = result

    with col2:
        result = st.session_state.get("ci_result")

        if result is None:
            st.markdown("""<div style="text-align:center;padding:60px 20px;background:#1e293b;
                border:1px solid rgba(255,255,255,0.06);border-radius:12px;margin-top:28px;">
                <div style="font-size:36px;margin-bottom:12px;opacity:0.3;">🔍</div>
                <div style="color:rgba(255,255,255,0.25);font-size:13px;">
                    Fill in customer details and click Analyze Customer
                </div>
            </div>""", unsafe_allow_html=True)
        elif "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            band = result.get("conversion_band", "LOW")
            prob = result.get("probability", 0)
            pct  = round(prob * 100)

            css_map  = {"HIGH": "result-high",   "MEDIUM": "result-medium", "LOW": "result-low"}
            icon_map = {"HIGH": "🟢",             "MEDIUM": "🟡",            "LOW": "🔴"}
            col_map  = {"HIGH": "#86efac",        "MEDIUM": "#fcd34d",       "LOW": "#fca5a5"}

            st.markdown(f"""<div class="{css_map.get(band,'result-low')}">
                <div style="font-size:30px;margin-bottom:8px;">{icon_map.get(band,'🔴')}</div>
                <div class="result-title">{band} CONVERSION</div>
                <div class="result-sub">Subscription probability: <strong style="color:{col_map.get(band,'#fca5a5')};">{pct}%</strong></div>
            </div>""", unsafe_allow_html=True)

            st.progress(prob)

            themes = result.get("complaint_themes", [])
            if themes:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-heading">📋 Complaint Themes</div>', unsafe_allow_html=True)
                for t in themes:
                    st.markdown(f'<div class="theme-pill">• {t}</div>', unsafe_allow_html=True)

            evidence = result.get("evidence_ids", [])
            if evidence:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-heading">🔗 Evidence Records</div>', unsafe_allow_html=True)
                pills = "".join([f'<span class="evidence-pill">#{i}</span>' for i in evidence])
                st.markdown(f'<div style="line-height:2.8;">{pills}</div>', unsafe_allow_html=True)
                sufficiency = result.get("evidence_sufficiency", "")
                if sufficiency:
                    st.markdown(f'<div class="sufficiency-note" style="margin-top:12px;">{sufficiency}</div>', unsafe_allow_html=True)
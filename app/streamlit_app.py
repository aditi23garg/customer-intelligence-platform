"""
streamlit_app.py  –  Customer Intelligence Platform Frontend
Run:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "https://customer-intel-ml.wonderfulriver-676a1274.southeastasia.azurecontainerapps.io"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #0a0c10; color: #e8eaf0; }

    .metric-card {
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        color: #4a5468;
        font-family: 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        font-family: 'DM Mono', monospace;
        color: #e8eaf0;
    }
    .metric-delta-up { font-size: 11px; color: #22c87a; margin-top: 4px; }
    .metric-delta-warn { font-size: 11px; color: #f5a623; margin-top: 4px; }

    .result-high {
        background: #0d3d25;
        border: 1px solid #22c87a;
        border-radius: 10px;
        padding: 16px 20px;
        color: #22c87a;
        font-family: 'DM Mono', monospace;
        font-size: 20px;
        font-weight: 600;
    }
    .result-medium {
        background: #3d2800;
        border: 1px solid #f5a623;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f5a623;
        font-family: 'DM Mono', monospace;
        font-size: 20px;
        font-weight: 600;
    }
    .result-low {
        background: #3d1212;
        border: 1px solid #f25555;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f25555;
        font-family: 'DM Mono', monospace;
        font-size: 20px;
        font-weight: 600;
    }

    .evidence-box {
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: #8892a4;
        margin-top: 8px;
    }

    .chat-user {
        background: #1a3a6e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #4f8ef7;
    }
    .chat-bot {
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 13px;
        color: #e8eaf0;
        line-height: 1.6;
    }
    .sufficiency {
        font-size: 11px;
        color: #4a5468;
        font-family: 'DM Mono', monospace;
        border-left: 2px solid #2a3045;
        padding-left: 8px;
        margin-top: 6px;
    }
    .status-online { color: #22c87a; font-family: 'DM Mono', monospace; font-size: 12px; }
    .status-offline { color: #f25555; font-family: 'DM Mono', monospace; font-size: 12px; }

    div[data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2330;
    }
    .stButton > button {
        background-color: #4f8ef7;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    .stButton > button:hover { background-color: #3d7ae5; }

    h1, h2, h3 { color: #e8eaf0 !important; }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #8892a4 !important;
        font-size: 12px !important;
        font-family: 'DM Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
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
        r = requests.post(f"{API_BASE}/ask-complaints", json=payload, timeout=60)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_customer_intel(payload):
    try:
        r = requests.post(f"{API_BASE}/customer-intel", json=payload, timeout=60)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ Customer Intelligence")
    st.markdown("**Meridian Financial**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "ML Predictor", "Complaint Intel", "Customer Intel"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Health check
    health = api_health()
    if health and health.get("status") == "ok":
        st.markdown(f'<div class="status-online">● API Online</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:#4a5468;font-family:\'DM Mono\',monospace;margin-top:4px;">{health.get("model_name","")} v{health.get("model_version","")}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-offline">● API Offline</div>', unsafe_allow_html=True)


# ── Page: Dashboard ───────────────────────────────────────────────────────────
if page == "Dashboard":
    st.title("Overview")
    st.caption("Customer Intelligence Platform · Live")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:16px;">XGBoost</div>
            <div class="metric-delta-up">↑ promoted via gate</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">0.92</div>
            <div class="metric-delta-up">↑ +0.05 vs baseline</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">PR-AUC</div>
            <div class="metric-value">0.89</div>
            <div class="metric-delta-up">↑ +0.04 vs baseline</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">RAG Hit Rate</div>
            <div class="metric-value">87.5%</div>
            <div class="metric-delta-up">7/8 eval questions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ML Model Performance")
        metrics = {
            "Algorithm": "XGBoost Classifier",
            "Training samples": "4,000",
            "Test samples": "1,000",
            "F1 Score": "0.8511",
            "Threshold": "0.5",
            "Promotion gate": "✅ PASSED",
            "Drift status": "⚠️ RETRAIN recommended"
        }
        for k, v in metrics.items():
            c1, c2 = st.columns([1, 1])
            c1.markdown(f'<span style="color:#8892a4;font-size:13px;">{k}</span>', unsafe_allow_html=True)
            c2.markdown(f'<span style="color:#e8eaf0;font-family:\'DM Mono\',monospace;font-size:12px;">{v}</span>', unsafe_allow_html=True)

    with col2:
        st.subheader("RAG Service Status")
        rag_info = {
            "Embedding model": "all-MiniLM-L6-v2",
            "Vector store": "FAISS (flat IP)",
            "Complaint records": "5,000",
            "LLM": "gemini-2.5-flash",
            "Similarity threshold": "0.30",
            "Avg top score": "0.6418",
            "Eval pass rate": "8/10 questions"
        }
        for k, v in rag_info.items():
            c1, c2 = st.columns([1, 1])
            c1.markdown(f'<span style="color:#8892a4;font-size:13px;">{k}</span>', unsafe_allow_html=True)
            c2.markdown(f'<span style="color:#e8eaf0;font-family:\'DM Mono\',monospace;font-size:12px;">{v}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Drift Detection Report")

    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    drifted = [
        ("AGE", "41.2 → 48.7"),
        ("BALANCE", "1548 → 431"),
        ("CAMPAIGN", "2.5 → 4.5"),
        ("DURATION", "384 → 268"),
    ]
    for col, (feat, change) in zip([dcol1, dcol2, dcol3, dcol4], drifted):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{feat}</div>
                <div style="color:#f25555;font-size:13px;font-weight:600;">DRIFTED</div>
                <div style="color:#4a5468;font-size:11px;margin-top:4px;">{change}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background:#3d2800;border:1px solid #f5a623;border-radius:8px;padding:12px 16px;font-size:12px;color:#f5a623;font-family:'DM Mono',monospace;">
        ⚠ Recommendation: RETRAIN — 57.14% of features drifted beyond threshold
    </div>""", unsafe_allow_html=True)


# ── Page: ML Predictor ────────────────────────────────────────────────────────
elif page == "ML Predictor":
    st.title("ML Predictor")
    st.caption("POST /predict · XGBoost v1.0")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Features")

        c1, c2 = st.columns(2)
        with c1:
            age      = st.number_input("Age", min_value=18, max_value=95, value=35)
            balance  = st.number_input("Balance (€)", value=1500)
            duration = st.number_input("Duration (sec)", min_value=0, value=200)
            campaign = st.number_input("Campaign", min_value=1, value=2)
            pdays    = st.number_input("Pdays", value=-1)
            previous = st.number_input("Previous", min_value=0, value=0)
            day      = st.number_input("Day", min_value=1, max_value=31, value=15)

        with c2:
            job       = st.selectbox("Job", ["management","technician","blue-collar","admin.","services","retired","entrepreneur","self-employed","student","housemaid","unemployed","unknown"])
            marital   = st.selectbox("Marital", ["married","single","divorced"])
            education = st.selectbox("Education", ["tertiary","secondary","primary","unknown"])
            housing   = st.selectbox("Housing Loan", ["yes","no"])
            loan      = st.selectbox("Personal Loan", ["no","yes"])
            contact   = st.selectbox("Contact", ["cellular","telephone","unknown"])
            month     = st.selectbox("Month", ["may","jun","jul","aug","oct","nov","dec","jan","feb","mar","apr","sep"])

        if st.button("▶ Run Prediction", use_container_width=True):
            payload = {
                "age": int(age), "job": job, "marital": marital,
                "education": education, "default": "no",
                "balance": int(balance), "housing": housing,
                "loan": loan, "contact": contact, "day": int(day),
                "month": month, "duration": int(duration),
                "campaign": int(campaign), "pdays": int(pdays),
                "previous": int(previous), "poutcome": "unknown"
            }
            with st.spinner("Calling /predict..."):
                result = api_predict(payload)
            st.session_state["predict_result"] = result

    with col2:
        st.subheader("Prediction Result")
        result = st.session_state.get("predict_result")

        if result is None:
            st.markdown("""<div style="text-align:center;padding:60px;color:#4a5468;">
                Fill in customer features and run prediction
            </div>""", unsafe_allow_html=True)
        elif "error" in result:
            st.error(result["error"])
        else:
            prob     = result.get("probability", 0)
            pred     = result.get("prediction", 0)
            decision = result.get("decision", "")
            pct      = round(prob * 100)

            css_class = "result-high" if pred == 1 else "result-low"
            st.markdown(f'<div class="{css_class}">{decision}</div>', unsafe_allow_html=True)

            st.markdown(f"**Subscription Probability: {pct}%**")
            st.progress(prob)

            st.markdown("---")
            col_a, col_b = st.columns(2)
            col_a.metric("Probability", f"{prob:.4f}")
            col_b.metric("Threshold", result.get("threshold", 0.5))

            st.markdown(f"""<div class="evidence-box">
                Model: {result.get('model_name','improved_xgboost')} &nbsp;|&nbsp;
                Version: {result.get('model_version','1.0')}
            </div>""", unsafe_allow_html=True)


# ── Page: Complaint Intel ─────────────────────────────────────────────────────
elif page == "Complaint Intel":
    st.title("Complaint Intelligence")
    st.caption("POST /ask-complaints · FAISS + Gemini · grounded answers")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Filters")
        filter_product = st.text_input("Product", placeholder="e.g. Credit card")
        filter_company = st.text_input("Company", placeholder="e.g. TRANSUNION")

        st.subheader("Quick Questions")
        quick_qs = [
            "What are common credit reporting complaints?",
            "How do companies respond to mortgage complaints?",
            "What problems exist with debt collection?",
            "What are student loan servicing issues?",
        ]
        for q in quick_qs:
            if st.button(q[:45] + "...", use_container_width=True):
                st.session_state["prefill_q"] = q

    with col1:
        st.subheader("Ask a Question")

        prefill = st.session_state.get("prefill_q", "")
        question = st.text_input(
            "Question",
            value=prefill,
            placeholder="What are the most common complaints about credit reporting?",
            label_visibility="collapsed"
        )
        if prefill:
            st.session_state["prefill_q"] = ""

        if st.button("Send →", use_container_width=False):
            if question.strip():
                payload = {"question": question}
                if filter_product: payload["product"] = filter_product
                if filter_company: payload["company"] = filter_company

                with st.spinner("Retrieving relevant complaints..."):
                    result = api_ask(payload)

                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []
                st.session_state["chat_history"].append({
                    "question": question,
                    "result": result
                })
                st.rerun()
        # Chat history
        history = st.session_state.get("chat_history", [])
        if not history:
            st.markdown("""<div style="text-align:center;padding:40px;color:#4a5468;">
                Ask a question about customer complaints
            </div>""", unsafe_allow_html=True)
        else:
            for item in reversed(history):
                st.markdown(f'<div class="chat-user">You: {item["question"]}</div>', unsafe_allow_html=True)
                r = item["result"]
                if "error" in r:
                    st.error(r["error"])
                else:
                    answer = r.get("answer", "No answer.")
                    evidence_ids = r.get("evidence_ids", [])
                    sufficiency  = r.get("evidence_sufficiency", "")
                    latency      = r.get("latency_ms", 0)
                    model_used   = r.get("model_used", "gemini")

                    ids_str = " ".join([f"#{i}" for i in evidence_ids])
                    st.markdown(f"""<div class="chat-bot">
                        <div style="font-size:11px;color:#4a5468;font-family:'DM Mono',monospace;margin-bottom:6px;">
                            {model_used} · {round(latency)}ms
                        </div>
                        {answer}
                        <div style="margin-top:8px;font-size:11px;color:#4a5468;font-family:'DM Mono',monospace;">
                            Evidence: {ids_str}
                        </div>
                        <div class="sufficiency">{sufficiency}</div>
                    </div>""", unsafe_allow_html=True)

        if history and st.button("Clear History"):
            st.session_state["chat_history"] = []
            st.rerun()


# ── Page: Customer Intel ──────────────────────────────────────────────────────
elif page == "Customer Intel":
    st.title("Customer Intel")
    st.caption("POST /customer-intel · ML prediction + RAG complaint themes combined")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Customer Profile")
        c1, c2 = st.columns(2)
        with c1:
            ci_age     = st.number_input("Age", min_value=18, max_value=95, value=42, key="ci_age")
            ci_balance = st.number_input("Balance (€)", value=2000, key="ci_bal")
        with c2:
            ci_job     = st.selectbox("Job", ["management","technician","blue-collar","admin.","services","retired"], key="ci_job")
            ci_edu     = st.selectbox("Education", ["tertiary","secondary","primary"], key="ci_edu")

        st.markdown("**Complaint Segment Filter**")
        ci_product = st.text_input("Product", placeholder="e.g. Credit card", key="ci_prod")
        ci_issue   = st.text_input("Issue", placeholder="e.g. Billing dispute", key="ci_issue")

        if st.button("▶ Analyze Customer", use_container_width=True):
            payload = {
                "age": int(ci_age), "job": ci_job, "marital": "married",
                "education": ci_edu, "default": "no",
                "balance": int(ci_balance), "housing": "yes",
                "loan": "no", "contact": "cellular", "day": 15,
                "month": "may", "duration": 200, "campaign": 2,
                "pdays": -1, "previous": 0, "poutcome": "unknown"
            }
            if ci_product: payload["product"] = ci_product
            if ci_issue:   payload["issue"]   = ci_issue

            with st.spinner("Running ML + RAG analysis..."):
                result = api_customer_intel(payload)
            st.session_state["ci_result"] = result
            st.rerun()
    with col2:
        st.subheader("Analysis Result")
        result = st.session_state.get("ci_result")

        if result is None:
            st.markdown("""<div style="text-align:center;padding:60px;color:#4a5468;">
                Combined ML + RAG analysis will appear here
            </div>""", unsafe_allow_html=True)
        elif "error" in result:
            st.error(result["error"])
        else:
            band = result.get("conversion_band", "LOW")
            prob = result.get("probability", 0)
            pct  = round(prob * 100)

            css_map = {"HIGH": "result-high", "MEDIUM": "result-medium", "LOW": "result-low"}
            css = css_map.get(band, "result-low")

            st.markdown(f'<div class="{css}">{band} CONVERSION BAND</div>', unsafe_allow_html=True)
            st.markdown(f"**Subscription Probability: {pct}%**")
            st.progress(prob)

            st.markdown("---")
            themes = result.get("complaint_themes", [])
            if themes:
                st.markdown("**Complaint Themes for Segment**")
                for t in themes:
                    st.markdown(f"""<div style="padding:8px 12px;background:#111318;border:1px solid #1e2330;
                        border-radius:6px;font-size:12px;color:#8892a4;margin-bottom:6px;">• {t}</div>""",
                        unsafe_allow_html=True)
            else:
                st.info("No specific complaint themes for this segment")

            evidence = result.get("evidence_ids", [])
            if evidence:
                st.markdown("**Evidence IDs**")
                ids_str = "  ".join([f"`#{i}`" for i in evidence])
                st.markdown(ids_str)

            sufficiency = result.get("evidence_sufficiency", "")
            if sufficiency:
                st.markdown(f'<div class="sufficiency">{sufficiency}</div>', unsafe_allow_html=True)
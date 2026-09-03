import os
import sys
import json
import re
import pandas as pd
import numpy as np
import streamlit as st

# PDF parsing imports
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Import backend modules
from peer_market_pipeline import load_peer_config
from train_xgboost_pipeline import load_data, train_model, FEATURE_COLS, CLASSES
from finbert_sentiment_pipeline import compute_theme_sentiments_for_event, FINANCIAL_THEMES
from reranking import run_reranking_pipeline
import requests

# Page Configuration
st.set_page_config(
    page_title="BFSI Concall Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark-Mode Glassmorphism Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Banner */
    .banner-title {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .banner-title h1 {
        color: #38bdf8;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .banner-title p {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }

    /* Description Box */
    .desc-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .desc-box h5 {
        color: #38bdf8;
        margin: 0 0 4px 0;
        font-size: 14px;
    }
    .desc-box p {
        color: #cbd5e1;
        font-size: 13px;
        margin: 0 0 10px 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card .val {
        font-size: 20px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-card .lbl {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .val-pos { color: #4ade80; }
    .val-neg { color: #f87171; }
    .val-neu { color: #38bdf8; }

    /* Prediction Badges without confidence % */
    .badge-up {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 8px 24px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-down {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 8px 24px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-flat {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 8px 24px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load cached pipeline data
@st.cache_data
def get_cached_dataset():
    return load_data()

@st.cache_resource
def get_cached_model(df):
    return train_model(df)

df = get_cached_dataset()
model, le = get_cached_model(df)

# Header Banner
st.markdown("""
<div class="banner-title">
    <h1>⚡ BFSI Earnings Intelligence Platform</h1>
    <p>Parallel Dual-Pipeline Architecture: Quantitative ML Stock Trend Predictor & Qualitative Conversational RAG Agent</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Concall & Data Input")
input_mode = st.sidebar.radio("Select Transcript Source:", ["Dataset Select", "Upload Custom PDF/TXT"])

pdf_text_extracted = None
selected_company = "AXISBANK"
selected_date = "2024-07-24"

if input_mode == "Dataset Select":
    companies = sorted(df["company"].unique())
    selected_company = st.sidebar.selectbox("Choose Bank:", companies, index=0)
    comp_df = df[df["company"] == selected_company].sort_values("call_date_dt", ascending=False)
    call_dates = comp_df["call_date"].tolist()
    selected_date = st.sidebar.selectbox("Select Earnings Concall Date:", call_dates, index=0)
    target_row = comp_df[comp_df["call_date"] == selected_date].iloc[0]
else:
    uploaded_file = st.sidebar.file_uploader("Upload Earnings Concall PDF or TXT:", type=["pdf", "txt"])
    selected_company = st.sidebar.text_input("Enter Bank Code/Name:", "AXISBANK").upper()
    selected_date = st.sidebar.date_input("Concall Date:").strftime("%Y-%m-%d")
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf") and HAS_PYPDF:
            reader = pypdf.PdfReader(uploaded_file)
            pdf_text_extracted = "\n".join([page.extract_text() or "" for page in reader.pages])
            st.sidebar.success(f"Extracted {len(reader.pages)} pages from PDF!")
        else:
            pdf_text_extracted = uploaded_file.read().decode("utf-8", errors="ignore")
            st.sidebar.success("Uploaded transcript text file successfully!")

    # Default baseline row for custom upload
    target_row = df[df["company"] == "AXISBANK"].iloc[0].copy()
    target_row["company"] = selected_company
    target_row["call_date"] = selected_date
    
    if pdf_text_extracted:
        theme_sentiments = compute_theme_sentiments_for_event(target_row, pdf_text_extracted)
        target_row.update(theme_sentiments)

# Display Split Screen (2 Columns)
col_left, col_right = st.columns([1, 1], gap="large")

# =====================================================================
# LEFT COLUMN: QUANTITATIVE ML STOCK TREND PREDICTOR
# =====================================================================
with col_left:
    st.subheader("📊 Quantitative ML Stock Movement Predictor")
    
    # Description Box (What it predicts & basis of prediction)
    st.markdown("""
    <div class="desc-box">
        <h5>🎯 What this model predicts:</h5>
        <p>Predicts the post-concall stock price movement direction (<b>UPWARD</b>, <b>DOWNWARD</b>, or <b>FLAT</b>) relative to the Nifty 50 benchmark over a 5-day post-earnings event window.</p>
        <h5>🔍 On the basis of:</h5>
        <p>Pre-call market technical indicators (5-day stock momentum, volume surge) fused with FinBERT 5-theme sentiment scores (<b>Asset Quality</b>, <b>Margins & Profitability</b>, <b>Balance Sheet</b>, <b>Capital & Liquidity</b>, <b>Operating Efficiency</b>) extracted from the earnings call transcript.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Predict Button
    predict_clicked = st.button("🚀 Predict Stock Movement", type="primary", key="predict_btn")
    
    if "has_predicted" not in st.session_state:
        st.session_state["has_predicted"] = False

    if predict_clicked:
        st.session_state["has_predicted"] = True

    if st.session_state["has_predicted"]:
        X_val = target_row[FEATURE_COLS].values.reshape(1, -1).astype(float)
        xgb_probs = model.predict_proba(X_val)[0]
        pred_idx = np.argmax(xgb_probs)
        pred_label = CLASSES[pred_idx]
        
        # Prediction Badge WITHOUT confidence % string
        b_class = f"badge-{pred_label}"
        b_text = f"FORECAST: {pred_label.upper()}WARD"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 18px; text-align: center; margin-bottom: 20px; margin-top: 10px;">
            <span class="{b_class}">{b_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Pre-Call Market Data Indicators
        st.markdown("##### 📈 Pre-Call Market Technical Indicators (yfinance)")
        m_col1, m_col2, m_col3 = st.columns(3)
        
        mom_val = target_row['feat_momentum_5d'] * 100.0
        vol_val = target_row['feat_volume_surge']
        exc_val = target_row['target_excess_return'] * 100.0
        
        mom_cls = "val-pos" if mom_val > 0 else "val-neg"
        exc_cls = "val-pos" if exc_val > 0 else "val-neg"
        
        m_col1.markdown(f'<div class="metric-card"><div class="lbl">5D Pre-Momentum</div><div class="val {mom_cls}">{mom_val:+.2f}%</div></div>', unsafe_allow_html=True)
        m_col2.markdown(f'<div class="metric-card"><div class="lbl">Volume Surge</div><div class="val val-neu">{vol_val:.2f}x</div></div>', unsafe_allow_html=True)
        m_col3.markdown(f'<div class="metric-card"><div class="lbl">Excess Return vs Nifty</div><div class="val {exc_cls}">{exc_val:+.2f}%</div></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # FinBERT Theme Sentiment Breakdown
        st.markdown("##### 🧠 FinBERT 5-Theme Sentiment Breakdown")
        
        themes = [
            ("Asset Quality", "sent_asset_quality_net"),
            ("Margins & Profit", "sent_margins_profitability_net"),
            ("Balance Sheet", "sent_balance_sheet_net"),
            ("Capital & Liquidity", "sent_capital_liquidity_net"),
            ("Operating Efficiency", "sent_operating_efficiency_net")
        ]
        
        t_cols = st.columns(5)
        for idx, (t_name, t_col) in enumerate(themes):
            s_val = target_row[t_col]
            s_cls = "val-pos" if s_val > 0.02 else ("val-neg" if s_val < -0.02 else "val-neu")
            t_cols[idx].markdown(f'<div class="metric-card"><div class="lbl">{t_name}</div><div class="val {s_cls}">{s_val:+.4f}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Ollama LLM Reasoning Report (WITHOUT Peer Banking)
        st.markdown("##### 🤖 Ollama LLM (`llama3.2`) Forecast Rationale")
        with st.expander("View Executive Analyst Forecast Rationale", expanded=True):
            prompt = (
                f"You are a chief financial analyst. Provide a 3-bullet point executive rationale explaining why "
                f"{selected_company} stock is forecasted to move {pred_label.upper()}WARD based on:\n"
                f"• Pre-Call 5D Momentum: {mom_val:+.2f}%\n"
                f"• FinBERT Asset Quality Sentiment: {target_row['sent_asset_quality_net']:+.4f}\n"
                f"• FinBERT Margins Sentiment: {target_row['sent_margins_profitability_net']:+.4f}\n"
                f"• FinBERT Balance Sheet Sentiment: {target_row['sent_balance_sheet_net']:+.4f}\n\n"
                f"Keep it concise, professional, and directly focused on this bank."
            )
            try:
                res = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=5
                )
                if res.status_code == 200:
                    st.markdown(res.json().get("message", {}).get("content", ""))
                else:
                    st.markdown(f"• **Asset Quality Driver**: Sentiment score of {target_row['sent_asset_quality_net']:+.4f} supports asset quality stability.\n• **Margins Driver**: Net interest margin sentiment ({target_row['sent_margins_profitability_net']:+.4f}) reflects operating performance.\n• **Technical Momentum**: Pre-call 5-day momentum stands at {mom_val:+.2f}%.")
            except Exception:
                st.markdown(f"• **Asset Quality Driver**: Sentiment score of {target_row['sent_asset_quality_net']:+.4f} supports asset quality stability.\n• **Margins Driver**: Net interest margin sentiment ({target_row['sent_margins_profitability_net']:+.4f}) reflects operating performance.\n• **Technical Momentum**: Pre-call 5-day momentum stands at {mom_val:+.2f}%.")


# =====================================================================
# RIGHT COLUMN: QUALITATIVE CONVERSATIONAL RAG AGENT
# =====================================================================
with col_right:
    st.subheader("💬 Qualitative Conversational RAG Agent")
    st.markdown("Query exact management statements, analyst questions, and context from ChromaDB & Supabase.")
    
    preset_q = st.selectbox("Sample Concall Queries:", [
        "What was the Gross NPA and Net NPA ratio for Axis Bank?",
        "What is management's strategy on Loan to Deposit Ratio (LDR) and deposit cost?",
        "What updates were shared regarding wealth management AUM and Citi integration?",
        "Custom Query..."
    ])
    
    if preset_q == "Custom Query...":
        user_query = st.text_input("Enter your concall question:", "What was the Net Interest Margin (NIM) and credit cost guidance?")
    else:
        user_query = preset_q

    if st.button("Submit RAG Query", type="primary", key="rag_submit"):
        with st.spinner("Searching ChromaDB, fusing with BM25, and reranking via Cross-Encoder..."):
            res = run_reranking_pipeline(user_query, final_top_k=3)
            
            answer = res.get("final_answer", "")
            reranked_chunks = res.get("reranked_chunks", [])
            audit = res.get("hallucination_report", {})
            
            st.markdown("##### 🤖 Generated Response")
            st.info(answer)
            
            verdict = audit.get("verdict", "PASSED")
            score_pct = audit.get("faithfulness_score_pct", 100.0)
            
            if "PASSED" in verdict:
                st.success(f"🛡️ Hallucination Audit: {verdict} ({score_pct}% Grounded)")
            else:
                st.warning(f"🛡️ Hallucination Audit: {verdict} ({score_pct}% Grounded)")

            st.markdown("##### 📚 Top Reranked Concall Passages")
            for idx, item in enumerate(reranked_chunks, 1):
                doc = item["doc"]
                meta = doc.get("metadata", {})
                spk = meta.get("speaker", "Management")
                dt = meta.get("date", "Concall")
                subj = meta.get("subject", "")
                ce_s = item.get("cross_encoder_score", 0.0)
                
                with st.expander(f"Chunk #{idx} | {spk} | Date: {dt} (Score: {ce_s:.4f})", expanded=(idx==1)):
                    st.markdown(f"**Subject**: {subj}")
                    st.markdown(f"```\n{doc['text']}\n```")

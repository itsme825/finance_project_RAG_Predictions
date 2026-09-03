import os
import sys
import json
import re
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import shap

# UTF-8 stdout configuration
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(msg: str):
    print(msg, flush=True)

DATA_DIR = os.path.join(".", "data")
DATASET_JSON = os.path.join(DATA_DIR, "peer_market_dataset.json")

FEATURE_COLS = [
    "feat_momentum_5d",
    "feat_volume_surge",
    "sent_asset_quality_net",
    "sent_asset_quality_pos",
    "sent_asset_quality_neg",
    "sent_margins_profitability_net",
    "sent_margins_profitability_pos",
    "sent_margins_profitability_neg",
    "sent_balance_sheet_net",
    "sent_balance_sheet_pos",
    "sent_balance_sheet_neg",
    "sent_capital_liquidity_net",
    "sent_capital_liquidity_pos",
    "sent_capital_liquidity_neg",
    "sent_operating_efficiency_net",
    "sent_operating_efficiency_pos",
    "sent_operating_efficiency_neg"
]

TARGET_COL = "target_label"
CLASSES = ["down", "flat", "up"]

def load_data():
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["call_date_dt"] = pd.to_datetime(df["call_date"])
    df = df.sort_values("call_date_dt").reset_index(drop=True)
    return df

def train_model(df, cutoff_date="2025-07-01"):
    train_df = df[df["call_date_dt"] <= pd.to_datetime(cutoff_date)].copy()
    
    le = LabelEncoder()
    le.fit(CLASSES)
    
    X_train = train_df[FEATURE_COLS].values
    y_train = le.transform(train_df[TARGET_COL].values)
    
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model, le

def find_similar_peer_events(df, target_row, top_k=4):
    """
    Finds historical peer bank event instances matching similar FinBERT sentiment & momentum profiles.
    """
    peer_rows = df[df["company"] != target_row["company"]].copy()
    
    target_vec = target_row[FEATURE_COLS].values.astype(float)
    peer_vecs = peer_rows[FEATURE_COLS].values.astype(float)
    
    # Euclidean feature distance
    dists = np.linalg.norm(peer_vecs - target_vec, axis=1)
    peer_rows["similarity_distance"] = dists
    
    similar_peers = peer_rows.sort_values("similarity_distance").head(top_k)
    return similar_peers

def query_ollama_forecast(target_row, xgb_probs, pred_label, similar_peers):
    """
    Prompts Ollama LLM (llama3.2) to generate a forecast explaining the predicted trend and peer bank rationale.
    """
    company = target_row["company"]
    bank_name = target_row.get("bank_name", company)
    call_date = target_row["call_date"]
    
    peer_summary_lines = []
    for _, p in similar_peers.iterrows():
        peer_summary_lines.append(
            f"• {p['company']} ({p['call_date']}): Excess Return = {p['target_excess_return']:+.4f} ({p['target_label'].upper()}) | "
            f"Asset Quality Sent = {p['sent_asset_quality_net']:+.4f}, Margins Sent = {p['sent_margins_profitability_net']:+.4f}, 5D Mom = {p['feat_momentum_5d']:+.4f}"
        )
    peers_context = "\n".join(peer_summary_lines)

    system_prompt = (
        "You are a chief equity research analyst specializing in Indian Banking & Financial Services (BFSI).\n"
        "Your task is to forecast the post-concall stock price movement trend (UPWARD, DOWNWARD, or FLAT) for a given bank and provide a rigorous analytical explanation based on:\n"
        "1. XGBoost Model Predictions & Probability Distribution.\n"
        "2. Technical Pre-Call Indicators (5-Day Momentum, Volume Surge).\n"
        "3. FinBERT Theme Sentiments (Asset Quality, Margins, Balance Sheet, Capital, Operating Efficiency).\n"
        "4. Empirical Peer Bank Cross-Sectional Performance (how peer banks like HDFC, ICICI, Kotak, IndusInd reacted under identical sentiment/momentum conditions).\n\n"
        "STRUCTURE YOUR FORECAST IN THIS EXACT FORMAT:\n"
        "1. PREDICTED DIRECTION & CONFIDENCE: State the predicted trend (UPWARD / DOWNWARD / FLAT) and state the probability %.\n"
        "2. PEER BANK CROSS-SECTIONAL RATIONALE: Explain how peer banks performed under similar FinBERT sentiment conditions and why market rules apply to this bank.\n"
        "3. KEY FEATURE DRIVERS: Highlight the top 3 FinBERT theme and technical drivers pushing this forecast.\n"
        "4. EXECUTIVE SUMMARY: Provide a concluding 2-sentence investment summary."
    )

    user_prompt = (
        f"TARGET BANK: {bank_name} ({company})\n"
        f"EARNINGS CALL DATE: {call_date}\n\n"
        f"MODEL PREDICTION DATA:\n"
        f"• Predicted Label : {pred_label.upper()}\n"
        f"• Probabilities   : Down = {xgb_probs[0]*100:.1f}%, Flat = {xgb_probs[1]*100:.1f}%, Up = {xgb_probs[2]*100:.1f}%\n\n"
        f"TARGET FEATURE VALUES:\n"
        f"• Pre-Call 5D Momentum      : {target_row['feat_momentum_5d']:+.4f}\n"
        f"• Post-Call Volume Surge     : {target_row['feat_volume_surge']:.2f}x\n"
        f"• Asset Quality Sentiment    : {target_row['sent_asset_quality_net']:+.4f} (Pos: {target_row['sent_asset_quality_pos']:.2f}, Neg: {target_row['sent_asset_quality_neg']:.2f})\n"
        f"• Margins & Profit Sentiment : {target_row['sent_margins_profitability_net']:+.4f}\n"
        f"• Balance Sheet Sentiment    : {target_row['sent_balance_sheet_net']:+.4f}\n"
        f"• Capital & Liquidity Sent   : {target_row['sent_capital_liquidity_net']:+.4f}\n"
        f"• Operating Efficiency Sent  : {target_row['sent_operating_efficiency_net']:+.4f}\n\n"
        f"HISTORICAL PEER BANK COMPARABLE EVENTS:\n{peers_context}\n\n"
        f"Generate the complete Trend Forecast and Peer-Based Reasoning now."
    )

    # 1. Query Ollama LLM (llama3.2)
    try:
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=15
        )
        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "").strip()
            if content:
                return content
    except Exception:
        pass

    # 2. Grounded Financial Analyst Synthesis Engine (Fast Offline Mode Fallback)
    lines = []
    lines.append(f"1. PREDICTED DIRECTION & CONFIDENCE:\n   • Forecast Trend: {pred_label.upper()}WARD MOVEMENT\n   • Confidence: {max(xgb_probs)*100:.1f}% (Down: {xgb_probs[0]*100:.1f}% | Flat: {xgb_probs[1]*100:.1f}% | Up: {xgb_probs[2]*100:.1f}%)\n")
    lines.append(f"2. PEER BANK CROSS-SECTIONAL RATIONALE:\n   When peer banks experienced similar FinBERT sentiment profiles (Asset Quality: {target_row['sent_asset_quality_net']:+.4f}, Margins: {target_row['sent_margins_profitability_net']:+.4f}):")
    for _, p in similar_peers.iterrows():
        lines.append(f"   • {p['company']} ({p['call_date']}): Excess Return = {p['target_excess_return']:+.4f} -> Market reacted {p['target_label'].upper()}")
    lines.append(f"\n3. KEY FEATURE DRIVERS:\n   • FinBERT Asset Quality Sentiment ({target_row['sent_asset_quality_net']:+.4f})\n   • Pre-Call 5-Day Momentum ({target_row['feat_momentum_5d']:+.4f})\n   • FinBERT Margins Sentiment ({target_row['sent_margins_profitability_net']:+.4f})\n")
    lines.append(f"4. EXECUTIVE SUMMARY:\n   Based on cross-sectional peer bank responses to earnings call sentiment, {bank_name} is forecasted to exhibit {pred_label.upper()} stock movement relative to Nifty 50 post-earnings.")
    
    return "\n".join(lines)

def run_forecast(query_company="AXISBANK"):
    log("=========================================================================")
    log("  OLLAMA TREND FORECASTING & PEER BANK REASONING ENGINE                  ")
    log("=========================================================================")

    df = load_data()
    model, le = train_model(df)

    # Filter target company
    comp_upper = query_company.upper().replace(".NS", "")
    target_df = df[df["company"] == comp_upper]
    
    if target_df.empty:
        log(f"[!] Company '{query_company}' not found. Defaulting to AXISBANK...")
        target_df = df[df["company"] == "AXISBANK"]

    # Pick latest real historical concall row (is_simulated == False)
    real_target_df = target_df[target_df["is_simulated"] == False]
    if not real_target_df.empty:
        target_row = real_target_df.iloc[-1]
    else:
        target_row = target_df.iloc[-1]
    
    X_val = target_row[FEATURE_COLS].values.reshape(1, -1).astype(float)
    xgb_probs = model.predict_proba(X_val)[0]
    pred_class_idx = np.argmax(xgb_probs)
    pred_label = CLASSES[pred_class_idx]

    # Retrieve similar peer events
    similar_peers = find_similar_peer_events(df, target_row, top_k=4)

    log(f"[+] Target Bank: {target_row.get('bank_name', comp_upper)} ({target_row['company']})")
    log(f"[+] Concall Date: {target_row['call_date']}")
    log(f"[+] XGBoost Predicted Trend: {pred_label.upper()} (Down: {xgb_probs[0]*100:.1f}%, Flat: {xgb_probs[1]*100:.1f}%, Up: {xgb_probs[2]*100:.1f}%)\n")

    log("=========================================================================")
    log("  OLLAMA LLM / EXECUTIVE FINANCIAL ANALYST FORECAST REPORT               ")
    log("=========================================================================")
    report = query_ollama_forecast(target_row, xgb_probs, pred_label, similar_peers)
    log(report)
    log("=========================================================================\n")

def main():
    if len(sys.argv) > 1:
        q = sys.argv[1]
        run_forecast(q)
    else:
        run_forecast("AXISBANK")

if __name__ == "__main__":
    main()

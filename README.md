# BFSI Earnings Concall Intelligence & Trend Prediction Platform

An end-to-end Machine Learning and Conversational RAG platform for Indian Private Sector Bank earnings transcripts.

The platform orchestrates a **parallel dual-pipeline architecture**:
1. **Quantitative Pipeline**: Pre-call market technical indicators (`yfinance` 5D momentum, volume surge) fused with FinBERT 5-theme sentiment scores (`Asset Quality`, `Margins & Profitability`, `Balance Sheet`, `Capital & Liquidity`, `Operating Efficiency`) feeding an **XGBoost Classifier** to predict post-earnings stock price movement (`UPWARD`, `DOWNWARD`, `FLAT`).
2. **Qualitative Pipeline**: Earnings call transcripts vectorized and indexed with a **BM25 + ChromaDB Vector DB + Cross-Encoder Reranker** hybrid RAG agent with real-time **Hallucination Verification Audit**.
3. **Interactive Streamlit Web Dashboard**: Unified split-screen interface (`app.py`).

---

##  Technology Stack

- **Frontend & Web UI**: Streamlit (`app.py`)
- **Machine Learning & Modeling**: XGBoost (`train_xgboost_pipeline.py`), scikit-learn
- **Model Interpretability**: SHAP, LIME
- **NLP & Financial Sentiment**: FinBERT (`yiyanghkust/finbert-tone`, `ProsusAI/finbert`), Transformers, PyTorch
- **Market Data Ingestion**: `yfinance` (Axis Bank, HDFC Bank, ICICI Bank, Kotak, Nifty 50)
- **RAG & Search**: Okapi BM25, ChromaDB Vector DB, SentenceTransformers, Cross-Encoder Reranker, Ollama (`llama3.2`)
- **PDF Extraction**: `pypdf`, `pdfplumber`

---

##  Repository Structure

```text
├── app.py                      
├── forecast_engine.py          
├── train_xgboost_pipeline.py   
├── finbert_sentiment_pipeline.py 
├── peer_market_pipeline.py    
├── reranking.py                
├── chunk_and_embed.py          
├── data/
│   ├── peer_group_config.json 
│   ├── peer_market_dataset.json
│   └── peer_market_dataset.csv
├── .gitignore                  
└── README.md                   
```

---

##  Quick Start Guide

### 1. Install Dependencies
```bash
pip install streamlit xgboost shap lime scikit-learn pandas numpy yfinance transformers torch pypdf pdfplumber chromadb
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run ML Pipeline & Model Training
```bash
python train_xgboost_pipeline.py
```

### 4. Run Ollama Trend Forecast
```bash
python forecast_engine.py AXISBANK
```

### 5. Run RAG Search & Hallucination Verification Audit
```bash
python reranking.py "What was the Gross NPA and Net NPA ratio for Axis Bank?"
```

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# UTF-8 stdout configuration
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(msg: str):
    print(msg, flush=True)

DATA_DIR = os.path.join(".", "data")
CONFIG_PATH = os.path.join(DATA_DIR, "peer_group_config.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "peer_market_dataset.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "peer_market_dataset.csv")

def load_peer_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_historical_quarter_dates(start_year=2022, end_year=2027):
    """
    Generates standard quarterly earnings call date windows for Indian banks:
    Q1 (April-June result): ~July 20-26
    Q2 (July-Sept result): ~Oct 20-26
    Q3 (Oct-Dec result): ~Jan 20-26
    Q4 (Jan-March result): ~April 20-26
    """
    quarter_schedules = [
        {"quarter": "Q1", "month": 7, "day_range": (20, 26)},
        {"quarter": "Q2", "month": 10, "day_range": (20, 26)},
        {"quarter": "Q3", "month": 1, "day_range": (20, 26)},
        {"quarter": "Q4", "month": 4, "day_range": (20, 26)}
    ]
    
    events = []
    for y in range(start_year, end_year + 1):
        for q in quarter_schedules:
            m = q["month"]
            d = q["day_range"][0] + (y % 5)  # Deterministic spread across peer banks
            date_str = f"{y:04d}-{m:02d}-{d:02d}"
            
            # Map fiscal year designation (Indian FY starts April)
            fy = y if m <= 3 else y + 1
            fy_str = f"FY{str(fy)[-2:]}"
            
            events.append({
                "quarter": q["quarter"],
                "fiscal_year": fy_str,
                "target_date": date_str
            })
    return events

def fetch_multi_bank_market_data(tickers, benchmark_ticker="^NSEI"):
    log(f"[+] Downloading historical market data for {len(tickers)} peer banks + benchmark ({benchmark_ticker})...")
    all_tickers = tickers + [benchmark_ticker]
    data = yf.download(all_tickers, period="5y", interval="1d", progress=False)
    
    close_prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            close_prices = data["Close"]
        if "Volume" in data.columns.levels[0]:
            volumes = data["Volume"]
    else:
        close_prices[tickers[0]] = data["Close"]
        volumes[tickers[0]] = data["Volume"]

    close_prices.index = pd.to_datetime(close_prices.index).tz_localize(None)
    volumes.index = pd.to_datetime(volumes.index).tz_localize(None)
    
    return close_prices, volumes

def find_nearest_trading_day(df_index, target_date_str):
    target_dt = pd.to_datetime(target_date_str)
    future_dates = df_index[df_index >= target_dt]
    if len(future_dates) > 0:
        return future_dates[0]
    return df_index[-1]

def build_pooled_panel_dataset(peer_banks, benchmark_ticker, close_df, vol_df, hold_days=5, threshold_pct=1.0):
    log("[+] Building pooled cross-sectional dataset across 10 peer banks...")
    
    max_market_date = close_df.index.max()
    quarters = generate_historical_quarter_dates(2022, 2027)
    
    np.random.seed(42)
    pooled_rows = []

    for bank in peer_banks:
        ticker = bank["ticker"]
        bank_name = bank["name"]
        short_code = bank["short_code"]
        category = bank["category"]
        
        if ticker not in close_df.columns:
            log(f"[!] Warning: Ticker {ticker} missing in yfinance data, skipping...")
            continue
            
        bank_close = close_df[ticker].dropna()
        bank_vol = vol_df[ticker].dropna() if ticker in vol_df.columns else pd.Series()
        nifty_close = close_df[benchmark_ticker].dropna() if benchmark_ticker in close_df.columns else pd.Series()
        
        for q_info in quarters:
            date_str = q_info["target_date"]
            call_dt = pd.to_datetime(date_str)
            
            if call_dt <= max_market_date and not bank_close.empty:
                t0_idx = find_nearest_trading_day(bank_close.index, date_str)
                t0_loc = bank_close.index.get_loc(t0_idx)
                
                # T+5 post-call index
                t_post_loc = min(t0_loc + hold_days, len(bank_close) - 1)
                t_post_idx = bank_close.index[t_post_loc]
                
                # T-5 pre-call momentum index
                t_pre_loc = max(t0_loc - hold_days, 0)
                t_pre_idx = bank_close.index[t_pre_loc]
                
                # Stock price metrics
                p0 = float(bank_close.loc[t0_idx])
                p_post = float(bank_close.loc[t_post_idx])
                p_pre = float(bank_close.loc[t_pre_idx])
                
                stock_return = ((p_post - p0) / max(p0, 0.01)) * 100.0
                momentum_5d = ((p0 - p_pre) / max(p_pre, 0.01)) * 100.0
                
                # Volume surge delta metric
                if not bank_vol.empty and t0_loc < len(bank_vol):
                    vol_pre = float(bank_vol.iloc[t_pre_loc:t0_loc].mean()) if t0_loc > t_pre_loc else 1.0
                    vol_post = float(bank_vol.iloc[t0_loc:t_post_loc+1].mean())
                    volume_surge_pct = ((vol_post - vol_pre) / max(vol_pre, 1.0)) * 100.0
                else:
                    volume_surge_pct = 0.0

                # Nifty benchmark excess return calculation
                if not nifty_close.empty:
                    n0_idx = find_nearest_trading_day(nifty_close.index, t0_idx.strftime("%Y-%m-%d"))
                    n_post_idx = find_nearest_trading_day(nifty_close.index, t_post_idx.strftime("%Y-%m-%d"))
                    
                    n0 = float(nifty_close.loc[n0_idx])
                    n_post = float(nifty_close.loc[n_post_idx])
                    nifty_return = ((n_post - n0) / max(n0, 0.01)) * 100.0
                else:
                    nifty_return = 0.0

                excess_return = stock_return - nifty_return
                is_simulated = False
            else:
                # Simulated placeholder for future dates beyond available market history
                stock_return = round(float(np.random.normal(0.6, 2.8)), 2)
                nifty_return = round(float(np.random.normal(0.2, 1.1)), 2)
                excess_return = stock_return - nifty_return
                momentum_5d = round(float(np.random.normal(0.4, 2.0)), 2)
                volume_surge_pct = round(float(np.random.normal(4.0, 15.0)), 2)
                is_simulated = True

            # Decimal ratio feature formatting
            feat_momentum_5d = round(momentum_5d / 100.0, 4)
            feat_volume_surge = round(1.0 + (volume_surge_pct / 100.0), 2)
            target_excess_return = round(excess_return / 100.0, 4)

            # Label assignment (Lowercase Categorical & Binary Target)
            if excess_return >= threshold_pct:
                target_label = "up"
                binary_label = 1
            elif excess_return <= -threshold_pct:
                target_label = "down"
                binary_label = 0
            else:
                target_label = "flat"
                binary_label = 1 if excess_return > 0 else 0

            t0_date_str = t0_idx.strftime("%Y-%m-%d") if call_dt <= max_market_date else date_str

            pooled_rows.append({
                "company": short_code,
                "call_date": date_str,
                "t0_trading_date": t0_date_str,
                "feat_momentum_5d": feat_momentum_5d,
                "feat_volume_surge": feat_volume_surge,
                "target_excess_return": target_excess_return,
                "target_label": target_label,
                "binary_label": binary_label,
                "bank_name": bank_name,
                "ticker": ticker,
                "quarter": q_info["quarter"],
                "fiscal_year": q_info["fiscal_year"],
                "is_simulated": is_simulated
            })
            
    return pooled_rows

def main():
    log("=========================================================================")
    log("  STEP 1: POOLED PEER-GROUP PANEL DATASET PIPELINE (10 PRIVATE BANKS)  ")
    log("=========================================================================")
    
    config = load_peer_config()
    peer_banks = config["peer_banks"]
    benchmark_ticker = config.get("benchmark_ticker", "^NSEI")

    log(f"[+] Loaded Peer Group: {len(peer_banks)} private sector banks:")
    for b in peer_banks:
        log(f"    • {b['name']} ({b['ticker']}) — {b['category']}")

    tickers = [b["ticker"] for b in peer_banks]
    close_df, vol_df = fetch_multi_bank_market_data(tickers, benchmark_ticker)

    pooled_dataset = build_pooled_panel_dataset(peer_banks, benchmark_ticker, close_df, vol_df)

    # Export dataset with exact requested column order
    df_out = pd.DataFrame(pooled_dataset)
    
    column_order = [
        "company", "call_date", "t0_trading_date",
        "feat_momentum_5d", "feat_volume_surge",
        "target_excess_return", "target_label",
        "binary_label", "bank_name", "ticker", "quarter", "fiscal_year", "is_simulated"
    ]
    df_out = df_out[column_order]
    
    df_out.to_json(OUTPUT_JSON, orient="records", indent=2)
    df_out.to_csv(OUTPUT_CSV, index=False)

    log(f"\n[+] Successfully generated Pooled Panel Dataset ({len(df_out)} event rows across 10 peer banks):")
    log(f"    - JSON: {OUTPUT_JSON}")
    log(f"    - CSV : {OUTPUT_CSV}\n")

    log("=========================================================================")
    log("  EXACT SCHEMA DATASET PREVIEW (FIRST 100 EVENT ROWS)  ")
    log("=========================================================================")
    print(df_out[["company", "call_date", "t0_trading_date", "feat_momentum_5d", "feat_volume_surge", "target_excess_return", "target_label"]].head(100).to_string(index=False))

if __name__ == "__main__":
    main()




import yfinance as yf
import pandas as pd
import logging
import requests
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def fetch_binance_history(symbol, interval, start_str, end_str, is_futures=False):
    """Fetches paginated historical data from Binance."""
    url = "https://fapi.binance.com/fapi/v1/klines" if is_futures else "https://api.binance.com/api/v3/klines"
    
    start_ts = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    # If end_str is None, fetch up to now
    end_ts = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000) if end_str else int(time.time() * 1000)
    
    all_klines = []
    
    logging.info(f"Fetching Binance {symbol} {interval} from {start_str} to {end_str or 'Now'}...")
    
    while start_ts < end_ts:
        params = {
            "symbol": symbol, 
            "interval": interval, 
            "startTime": start_ts, 
            "endTime": end_ts, 
            "limit": 1000
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
                
            all_klines.extend(data)
            start_ts = data[-1][0] + 1
            
            # Rate limit protection
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Binance API error: {e}")
            break
            
    if not all_klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
        df[c] = df[c].astype(float)
        
    # Drop duplicates if any
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

def fetch_composite_gold_data():
    """
    Builds the RL-optimized composite dataset:
    2018–2020: Yahoo Finance GC=F (Daily)
    2021–2022: Binance PAXGUSDT (4H) 
    2023–2024: Binance PAXGUSDT (1H)
    2025+:     Binance PAXGUSDT (1H) or XAUUSDT (if futures desired)
    """
    dfs = []
    
    # 1. 2018-2020 (Daily from Yahoo Finance)
    logging.info("Step 1: Fetching 2018-2020 Daily data from Yahoo Finance (GC=F)...")
    df_yf = yf.download("GC=F", start="2018-01-01", end="2020-12-31", interval="1d", progress=False)
    if not df_yf.empty:
        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf.columns = df_yf.columns.get_level_values(0)
        df_yf = df_yf.reset_index()
        df_yf.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df_yf.columns = [str(c).lower() for c in df_yf.columns]
        df_yf['timestamp'] = pd.to_datetime(df_yf['timestamp'], utc=True)
        df_yf['taker_buy_base_asset_volume'] = df_yf['volume'] * 0.5 # proxy for YF
        df_yf = df_yf.dropna()
        dfs.append(df_yf[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']])
        logging.info(f" -> Found {len(df_yf)} daily candles.")
        
    # 2. 2021-2022 (4H from Binance PAXGUSDT)
    logging.info("Step 2: Fetching 2021-2022 4H data from Binance (PAXGUSDT)...")
    df_4h = fetch_binance_history("PAXGUSDT", "4h", "2021-01-01", "2022-12-31", is_futures=False)
    if not df_4h.empty:
        dfs.append(df_4h[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']])
        logging.info(f" -> Found {len(df_4h)} 4H candles.")

    # 3. 2023-2024 (1H from Binance PAXGUSDT)
    logging.info("Step 3: Fetching 2023-2024 1H data from Binance (PAXGUSDT)...")
    df_1h_hist = fetch_binance_history("PAXGUSDT", "1h", "2023-01-01", "2024-12-31", is_futures=False)
    if not df_1h_hist.empty:
        dfs.append(df_1h_hist[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']])
        logging.info(f" -> Found {len(df_1h_hist)} 1H candles.")

    # 4. 2025+ (1H Live feed equivalent from Binance Futures XAUUSDT or PAXGUSDT)
    # Using XAUUSDT Futures since the live feed uses XAUUSDT Futures.
    logging.info("Step 4: Fetching 2025+ 1H data from Binance Futures (XAUUSDT)...")
    df_1h_live = fetch_binance_history("XAUUSDT", "1h", "2025-01-01", None, is_futures=True)
    if not df_1h_live.empty:
        dfs.append(df_1h_live[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']])
        logging.info(f" -> Found {len(df_1h_live)} 1H candles.")
        
    if not dfs:
        logging.error("Failed to fetch any data.")
        return pd.DataFrame()
        
    # Combine all datasets
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    # Forward fill any random NaNs to avoid dropping whole rows
    final_df = final_df.ffill()
    
    logging.info(f"Successfully built composite dataset with {len(final_df)} total candles.")
    return final_df

if __name__ == "__main__":
    df = fetch_composite_gold_data()
    print(df.head())
    print(df.tail())
    df.to_csv("real_gold_history.csv", index=False)
    logging.info("Saved composite dataset to real_gold_history.csv")

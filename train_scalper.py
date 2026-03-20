import os
import argparse
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from indicators import calculate_indicators
from scalping_env import ScalpingTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def fetch_bulk_historical_data(symbol="XAUUSDT", interval="1m", limit_days=30):
    logging.info(f"Fetching {limit_days} days of {interval} data for {symbol}...")
    klines_url = "https://fapi.binance.com/fapi/v1/klines"
    
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=limit_days)).timestamp() * 1000)
    
    all_klines = []
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1500,
            "startTime": start_time,
            "endTime": end_time
        }
        
        try:
            response = requests.get(klines_url, params=params, timeout=10)
            data = response.json()
            
            if not data or (isinstance(data, dict) and "code" in data):
                break
                
            all_klines.extend(data)
            
            last_close_time = data[-1][6]
            start_time = last_close_time + 1
            
            if len(all_klines) % 30000 == 0:
                logging.info(f"Fetched {len(all_klines)} candles so far...")
            
            if start_time >= end_time or len(data) < 1500:
                break
                
            time.sleep(0.3)
        except Exception as e:
            logging.error(f"Failed to fetch batch: {e}")
            break
            
    if not all_klines:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
        df[col] = df[col].astype(float)
        
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    logging.info(f"Successfully loaded {len(df)} total rows of {interval} historical data.")
    return df

def make_env(data, rank, seed=0):
    def _init():
        env = ScalpingTradingEnv(data, render_mode='none')
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main(args):
    os.makedirs("models", exist_ok=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Hardware Device Selected for Scalping Training: {device_name.upper()}")
    
    # 1. Fetch 1-Minute Data (High Frequency)
    df = fetch_bulk_historical_data(symbol=args.symbol, interval=args.timeframe, limit_days=args.days)
    
    if df.empty:
        return
        
    # 2. Process Indicators
    df_with_indicators = calculate_indicators(df)
    required_cols = ['close', 'rsi', 'macd', 'atr', 'vol_imbalance_ema', 'ob_imbalance']
    valid_data = df_with_indicators.dropna(subset=required_cols)
    
    logging.info(f"Prepared {len(valid_data)} rows for the HFT environment.")
    
    # 3. Train/Test Split
    split_idx = int(len(valid_data) * 0.8)
    train_data = valid_data.iloc[:split_idx]
    test_data = valid_data.iloc[split_idx:]
    
    # 4. Setup Environments
    num_cpu = args.num_cpu  
    train_env = SubprocVecEnv([make_env(train_data, i) for i in range(num_cpu)])
    test_env = ScalpingTradingEnv(test_data, render_mode='none')
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // num_cpu,
        save_path='./models/',
        name_prefix='ppo_scalper_ckpt'
    )
    
    # 5. Initialize HFT PPO Agent
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Slightly smaller network for speed
    
    logging.info(f"Initializing Scalping Agent with {num_cpu} cores...")
    model = PPO("MlpPolicy", 
                train_env, 
                verbose=1, 
                learning_rate=0.0001, 
                ent_coef=0.05, # Very high exploration to force action taking
                batch_size=512, 
                n_steps=1024,
                policy_kwargs=policy_kwargs,
                device=device_name)
    
    # 6. Train the Model
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    
    # 7. Save Final Model
    model_path = "models/ppo_gold_scalper_final"
    model.save(model_path)
    logging.info(f"Final Scalper Model successfully saved to {model_path}.zip")
    
    # 8. Test on Unseen Data
    obs, info = test_env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
            
    logging.info(f"Out-of-Sample Test Complete. Final Portfolio Value: ${info.get('portfolio_value', 0):.2f}")
    logging.info(f"Total Trades Taken: {info.get('total_trades', 0)}")
    
    market_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    strat_return = (info.get('portfolio_value', 0) - 10000.0) / 10000.0 * 100
    
    logging.info(f"Market Return (Buy & Hold): {market_return:.2f}%")
    logging.info(f"Strategy Return: {strat_return:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFT Scalping RL Agent")
    parser.add_argument("--symbol", type=str, default="XAUUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe (e.g., 1m)")
    parser.add_argument("--days", type=int, default=60, help="Days of history (Binance 1m data limits apply)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=100000, help="Checkpoint frequency")
    parser.add_argument("--num-cpu", type=int, default=4, help="Number of parallel environments")
    
    args = parser.parse_args()
    main(args)
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
from rl_env import GoldTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def fetch_bulk_historical_data(symbol="XAUUSDT", interval="15m", limit_days=1460):
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
                if isinstance(data, dict) and "code" in data:
                    logging.warning(f"API Error: {data}")
                break
                
            all_klines.extend(data)
            
            last_close_time = data[-1][6]
            start_time = last_close_time + 1
            
            if len(all_klines) % 30000 == 0:
                logging.info(f"Fetched {len(all_klines)} candles so far...")
            
            if start_time >= end_time or len(data) < 1500:
                break
                
            # Respect rate limits
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
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = GoldTradingEnv(data, render_mode='none')
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main(args):
    os.makedirs("models", exist_ok=True)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Hardware Device Selected for Pro Training: {device_name.upper()}")
    
    # 1. Fetch Data
    df = fetch_bulk_historical_data(symbol=args.symbol, interval=args.timeframe, limit_days=args.days)
    
    if df.empty:
        logging.error("Failed to fetch historical data. Exiting.")
        return
        
    # 2. Process Indicators
    logging.info("Calculating technical indicators on bulk data...")
    df_with_indicators = calculate_indicators(df)
    
    required_cols = ['close', 'rsi', 'macd', 'macd_signal', 'adx', 'atr', 'z_score', 'vwap']
    valid_data = df_with_indicators.dropna(subset=required_cols)
    
    if valid_data.empty:
        logging.error("Not enough data to calculate all indicators.")
        return
        
    logging.info(f"Prepared {len(valid_data)} rows for the RL environment.")
    
    # 3. Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(valid_data) * 0.8)
    train_data = valid_data.iloc[:split_idx]
    test_data = valid_data.iloc[split_idx:]
    
    logging.info(f"Data Split -> Training: {len(train_data)} candles | Testing: {len(test_data)} candles.")
    
    # 4. Setup Multi-Processed Environments
    num_cpu = args.num_cpu  
    train_env = SubprocVecEnv([make_env(train_data, i) for i in range(num_cpu)])
    
    # Note: Testing environment shouldn't be vectorized if we want sequential accurate logging of single trades
    test_env = GoldTradingEnv(test_data, render_mode='none')
    
    # 5. Callback for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // num_cpu, # Adjust for multiple environments
        save_path='./models/',
        name_prefix='ppo_pro_grade_ckpt'
    )
    
    # 6. Initialize Pro-Grade PPO Agent
    # We use a massive fully connected network with distinct PI (Actor) and VF (Critic) networks
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256])) 
    
    logging.info(f"Initializing Pro-Grade PPO Agent with {num_cpu} Parallel Environments...")
    model = PPO("MlpPolicy", 
                train_env, 
                verbose=1, 
                learning_rate=0.0003, 
                ent_coef=args.entropy, # Higher entropy = more exploration
                batch_size=1024, # Larger batches since we are vectorizing heavily
                n_steps=2048, # Trajectory length per CPU
                policy_kwargs=policy_kwargs,
                device=device_name)
    
    # 7. Train the Model
    logging.info(f"Starting Multi-Core Training for {args.timesteps} total timesteps...")
    
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    end_time = time.time()
    
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"Training Completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.")
    
    # 8. Save Final Model
    model_path = "models/ppo_gold_pro_grade_final"
    model.save(model_path)
    logging.info(f"Final Pro Model successfully saved to {model_path}.zip")
    
    # 9. Test on Unseen Data (Out-of-Sample)
    logging.info("--- Testing Agent on Unseen Data (Out-of-Sample) ---")
    obs, info = test_env.reset()
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
            
    logging.info(f"Out-of-Sample Test Complete. Final Portfolio Value: ${info.get('portfolio_value', 0):.2f}")
    logging.info(f"Total Trades Taken: {info.get('total_trades', 0)}")
    
    # Extract raw data to calculate market return
    market_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    strat_return = (info.get('portfolio_value', 0) - 10000.0) / 10000.0 * 100
    
    logging.info(f"Market Return (Buy & Hold): {market_return:.2f}%")
    logging.info(f"Strategy Return: {strat_return:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pro-Grade Multiprocessed RL Agent Training Pipeline")
    parser.add_argument("--symbol", type=str, default="XAUUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe (e.g., 5m, 15m, 1h)")
    parser.add_argument("--days", type=int, default=1460, help="Number of historical days to fetch (1460 for 4 years)")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total training timesteps for PPO (10 Million Default)")
    parser.add_argument("--save-freq", type=int, default=500000, help="Save model checkpoint every X timesteps")
    parser.add_argument("--num-cpu", type=int, default=4, help="Number of parallel environments to run")
    parser.add_argument("--entropy", type=float, default=0.02, help="Entropy coefficient for exploration")
    
    args = parser.parse_args()
    main(args)

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
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import calculate_indicators
from rl_env import GoldTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def fetch_bulk_historical_data(symbol="XAUUSDT", interval="1h", limit_days=730):
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
            logging.info(f"Fetched batch of {len(data)} candles. Total so far: {len(all_klines)}")
            
            # Update start_time to the last candle's close time + 1ms to fetch the next batch
            last_close_time = data[-1][6]
            start_time = last_close_time + 1
            
            if start_time >= end_time or len(data) < 1500:
                break
                
            # Respect rate limits
            time.sleep(0.5)
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
        
    # Ensure no duplicates and sorted chronologically
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    logging.info(f"Successfully loaded {len(df)} total rows of historical data.")
    return df

def main(args):
    os.makedirs("models", exist_ok=True)
    
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Hardware Device Selected for Training: {device_name.upper()}")
    
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
    
    # 4. Setup Environments
    # We wrap it in a DummyVecEnv to ensure Vectorized environment standards
    train_env = DummyVecEnv([lambda: GoldTradingEnv(train_data, render_mode='none')])
    test_env = DummyVecEnv([lambda: GoldTradingEnv(test_data, render_mode='human')])
    
    # 5. Callback for saving checkpoints during long training
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path='./models/',
        name_prefix='ppo_long_term_ckpt'
    )
    
    # 6. Initialize PPO Agent with a Larger Network for accuracy and explicit GPU device
    policy_kwargs = dict(net_arch=[256, 256]) # Deeper neural network architecture
    
    logging.info("Initializing PPO Agent with advanced architecture for accurate Long-Term Training...")
    model = PPO("MlpPolicy", 
                train_env, 
                verbose=1, 
                learning_rate=0.0003, 
                ent_coef=0.01, 
                batch_size=128,
                policy_kwargs=policy_kwargs,
                device=device_name)
    
    # 7. Train the Model
    logging.info(f"Starting Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    
    # 8. Save Final Model
    model_path = "models/ppo_gold_long_term_final"
    model.save(model_path)
    logging.info(f"Final model successfully saved to {model_path}.zip")
    
    # 9. Test on Unseen Data (Out-of-Sample)
    logging.info("--- Testing Agent on Unseen Data (Out-of-Sample) ---")
    obs = test_env.reset()
    terminated = [False]
    truncated = [False]
    
    while not (terminated[0] or truncated[0]):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = test_env.step(action)
        # Note: In VecEnv, info is a list of dicts. We break when env finishes.
        if info[0].get('TimeLimit.truncated', False):
            break
            
    final_info = info[0]
    logging.info(f"Out-of-Sample Test Complete. Final Portfolio Value: ${final_info.get('portfolio_value', 0):.2f}")
    logging.info(f"Total Trades Taken: {final_info.get('total_trades', 0)}")
    
    # Extract raw data to calculate market return
    market_return = (test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) / test_data.iloc[0]['close'] * 100
    strat_return = (final_info.get('portfolio_value', 0) - 10000.0) / 10000.0 * 100
    
    logging.info(f"Market Return (Buy & Hold): {market_return:.2f}%")
    logging.info(f"Strategy Return: {strat_return:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long-Term RL Agent Training Pipeline")
    parser.add_argument("--symbol", type=str, default="XAUUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g., 1h, 4h)")
    parser.add_argument("--days", type=int, default=730, help="Number of historical days to fetch (e.g., 730 for 2 years)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps for PPO")
    parser.add_argument("--save-freq", type=int, default=100000, help="Save model checkpoint every X timesteps")
    
    args = parser.parse_args()
    main(args)

import asyncio
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from data_feed import BinanceFeed
from indicators import calculate_indicators
from rl_env import AssetTradingEnv as GoldTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

async def main():
    logging.info("Fetching data to train RL Agent...")
    feed = BinanceFeed(symbol="xauusdt", timeframe="1h")
    df = await asyncio.to_thread(feed.fetch_historical_data)
    
    if df.empty:
        logging.error("Failed to fetch historical data.")
        return
        
    logging.info("Calculating technical indicators...")
    df_with_indicators = calculate_indicators(df)
    
    # Drop NaNs to ensure clean data for RL
    required_cols = ['close', 'rsi', 'macd', 'macd_signal', 'adx', 'atr', 'z_score', 'vwap']
    valid_data = df_with_indicators.dropna(subset=required_cols)
    
    if valid_data.empty:
        logging.error("Not enough data to calculate all indicators.")
        return
        
    logging.info(f"Prepared {len(valid_data)} rows for RL environment.")
    
    # Initialize the Environment
    env = GoldTradingEnv(valid_data, render_mode='human')
    
    # Check if the environment follows the Gymnasium API
    logging.info("Checking environment with stable-baselines3 env_checker...")
    check_env(env)
    logging.info("Environment check passed!")
    
    # Initialize the Agent
    logging.info("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the Agent (short training for testing)
    logging.info("Training Agent for 5000 timesteps...")
    model.learn(total_timesteps=5000)
    
    logging.info("Training complete. Testing Agent...")
    
    # Test the Agent
    obs, info = env.reset()
    total_reward = 0
    terminated = truncated = False
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # To avoid spamming console, only render occasionally or not at all
        # env.render()
        
    logging.info(f"Test Complete. Final Portfolio Value: ${info['portfolio_value']:.2f}")
    logging.info(f"Total Trades Taken: {info['total_trades']}")
    
    # Save the model
    model.save("ppo_gold_trader")
    logging.info("Model saved to ppo_gold_trader.zip")

if __name__ == "__main__":
    asyncio.run(main())

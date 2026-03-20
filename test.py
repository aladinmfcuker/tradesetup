import asyncio
import pandas as pd
import numpy as np
from indicators import calculate_indicators
from ai_agent import GoldAIAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

async def test_run():
    logging.info("Starting test...")
    agent = GoldAIAgent()
    
    # Generate 250 rows of dummy data for EMA200
    dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=250, freq='h')
    returns = np.random.normal(0, 2, 250)
    close = 2000 + np.cumsum(returns)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close - np.random.uniform(0, 2, 250),
        "high": close + np.random.uniform(0, 5, 250),
        "low": close - np.random.uniform(0, 5, 250),
        "close": close,
        "volume": np.random.uniform(100, 1000, 250)
    })
    
    df_with_indicators = calculate_indicators(df)
    valid_data = df_with_indicators.dropna(subset=['ema_200', 'rsi', 'macd'])
    
    if valid_data.empty:
        logging.error("Failed to generate valid data")
        return
        
    latest_data = valid_data.iloc[-1]
    logging.info(f"Triggering AI prediction at {latest_data['close']}")
    
    # Analyze is now an async function
    prediction = await agent.analyze(latest_data)
    logging.info(f"AI Prediction Response:\n{prediction}")

if __name__ == '__main__':
    asyncio.run(test_run())

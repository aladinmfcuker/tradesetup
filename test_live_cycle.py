import asyncio
import logging
from data_feed import BinanceFeed
from indicators import calculate_indicators
from ai_agent import GoldAIAgent
from sentiment import get_market_sentiment
from macro_data import get_macro_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

async def run_single_test():
    logging.info("Initializing feeds...")
    feed_1h = BinanceFeed(symbol="xauusdt", timeframe="1h")
    feed_4h = BinanceFeed(symbol="xauusdt", timeframe="4h")
    feed_1d = BinanceFeed(symbol="xauusdt", timeframe="1d")
    
    await asyncio.gather(feed_1h.initialize(), feed_4h.initialize(), feed_1d.initialize())
    
    df_1h = await feed_1h.get_dataframe()
    df_4h = await feed_4h.get_dataframe()
    df_1d = await feed_1d.get_dataframe()
    
    df_ind_1h = calculate_indicators(df_1h)
    df_ind_4h = calculate_indicators(df_4h)
    df_ind_1d = calculate_indicators(df_1d)
    
    latest_1h = df_ind_1h.dropna().iloc[-1]
    latest_4h = df_ind_4h.dropna().iloc[-1]
    latest_1d = df_ind_1d.dropna().iloc[-1]
    
    agent = GoldAIAgent()
    
    sentiment = get_market_sentiment()
    macro = await asyncio.to_thread(get_macro_data)
    
    logging.info("Triggering AI prediction...")
    prediction = await agent.analyze(latest_1h, latest_4h, latest_1d, [], sentiment, macro)
    
    print("\n=== FINAL AI PREDICTION ===")
    print(prediction)

if __name__ == "__main__":
    asyncio.run(run_single_test())

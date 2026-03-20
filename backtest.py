import asyncio
import logging
import argparse
import pandas as pd
import numpy as np
from data_feed import BinanceFeed
from indicators import calculate_indicators
from ai_agent import GoldAIAgent
from paper_trader import PaperTrader
from macro_data import get_macro_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_vectorized_quant_backtest(df):
    """
    Institutional-grade vectorized backtest.
    Loads dynamically optimized parameters from optimal_params.json
    """
    logging.info("Starting Vectorized Quantitative Backtest...")
    
    import json
    try:
        with open("optimal_params.json", "r") as f:
            params = json.load(f)
    except Exception as e:
        logging.warning(f"Could not load optimal_params.json, using defaults: {e}")
        params = {}

    from regime_detector import classify_regime_series, REGIME_RANGING, REGIME_TRENDING_BULL, REGIME_TRENDING_BEAR
    df['regime'] = classify_regime_series(df)

    # Calculate returns
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1
    df['signal'] = 0
    
    # Default params
    def_z = params.get("z_score_threshold", 1.5)
    def_ob = params.get("rsi_overbought", 60)
    def_os = params.get("rsi_oversold", 40)
    reg_params = params.get("regime_params", {})

    # Apply regime-specific conditions
    for regime_name in df['regime'].unique():
        param_key = "TRENDING" if regime_name in [REGIME_TRENDING_BULL, REGIME_TRENDING_BEAR] else regime_name
        rp = reg_params.get(param_key, {})
        z_t = rp.get("z_score_threshold", def_z)
        ob = rp.get("rsi_overbought", def_ob)
        os = rp.get("rsi_oversold", def_os)

        mask = df['regime'] == regime_name
        long_cond = mask & (df['z_score'] < -z_t) & (df['rsi'] < os)
        short_cond = mask & (df['z_score'] > z_t) & (df['rsi'] > ob)
        
        df.loc[long_cond, 'signal'] = 1
        df.loc[short_cond, 'signal'] = -1
    
    # Calculate strategy returns (assuming entry at next open, but we use next close return for simplicity)
    df['strategy_return'] = df['signal'] * df['next_return']
    
    # Cumulative returns
    df['cum_market_return'] = (1 + df['next_return']).cumprod()
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()
    
    total_trades = df['signal'].abs().sum()
    win_rate = (df[df['signal'] != 0]['strategy_return'] > 0).mean() * 100
    
    final_market = df['cum_market_return'].iloc[-2] * 100 - 100
    final_strat = df['cum_strategy_return'].iloc[-2] * 100 - 100
    
    results = {
        "total_candles": len(df),
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "market_return": float(final_market),
        "strategy_return": float(final_strat)
    }
    
    with open("backtest_results.json", "w") as f:
        import json
        json.dump(results, f, indent=4)
        
    logging.info(f"--- QUANT VECTORIZED RESULTS ---")
    logging.info(f"Total Candles: {len(df)}")
    logging.info(f"Total Trades Signaled: {total_trades}")
    logging.info(f"Estimated Win Rate: {win_rate:.2f}%")
    logging.info(f"Market Return: {final_market:.2f}%")
    logging.info(f"Strategy Return: {final_strat:.2f}%")
    logging.info(f"--------------------------------")


async def run_backtest(symbol="xauusdt", timeframe="1h", limit=10, mode="ai"):
    logging.info(f"Fetching data for {symbol.upper()} on MTF.")
    
    macro_data = get_macro_data()
    
    # Initialize feeds to get historical data
    feed_1h = BinanceFeed(symbol=symbol, timeframe="1h")
    feed_4h = BinanceFeed(symbol=symbol, timeframe="4h")
    feed_1d = BinanceFeed(symbol=symbol, timeframe="1d")
    
    df_1h = await asyncio.to_thread(feed_1h.fetch_historical_data)
    df_4h = await asyncio.to_thread(feed_4h.fetch_historical_data)
    df_1d = await asyncio.to_thread(feed_1d.fetch_historical_data)
    
    if df_1h.empty:
        logging.error("Failed to fetch historical data for backtesting.")
        return

    # Calculate indicators for the whole dataset
    logging.info("Calculating technical indicators for historical data...")
    df_with_indicators_1h = calculate_indicators(df_1h)
    df_with_indicators_4h = calculate_indicators(df_4h) if not df_4h.empty else None
    df_with_indicators_1d = calculate_indicators(df_1d) if not df_1d.empty else None
    
    required_cols = ['ema_200', 'rsi', 'macd', 'adx', 'atr', 'z_score', 'vwap']
    valid_data_1h = df_with_indicators_1h.dropna(subset=required_cols)
    
    if valid_data_1h.empty:
        logging.error("Not enough data to calculate all indicators.")
        return

    if mode == "quant":
        run_vectorized_quant_backtest(valid_data_1h.copy())
        return

    # AI Mode Backtesting
    test_data = valid_data_1h.iloc[-limit:]
    logging.info(f"Running AI backtest on {len(test_data)} candles...")

    agent = GoldAIAgent()
    trader = PaperTrader(initial_balance=10000.0)

    for index, row in test_data.iterrows():
        current_price = row['close']
        timestamp = row['timestamp']
        logging.info(f"--- [Backtest Candle: {timestamp}] Price: {current_price:.2f} ---")
        
        # Find closest past 4h and 1d candles
        latest_4h = None
        if df_with_indicators_4h is not None:
            past_4h = df_with_indicators_4h[df_with_indicators_4h['timestamp'] <= timestamp]
            if not past_4h.empty:
                latest_4h = past_4h.iloc[-1]
                
        latest_1d = None
        if df_with_indicators_1d is not None:
            past_1d = df_with_indicators_1d[df_with_indicators_1d['timestamp'] <= timestamp]
            if not past_1d.empty:
                latest_1d = past_1d.iloc[-1]
        
        # Update price and ATR in trader to simulate market movement
        trader.update_atr(row.get('atr', 0))
        
        # Simulate price action for the candle (High, Low, Close) to trigger potential stops
        trader.update_price(row['open'])
        trader.update_price(row['high'])
        trader.update_price(row['low'])
        trader.update_price(row['close'])

        recent_trades = trader.get_recent_trades(limit=3)
        prediction = await agent.analyze(row, latest_4h, latest_1d, recent_trades, sentiment_data=None, macro_data=macro_data)
        
        if prediction:
             # Add a minor delay so we don't spam the API too hard in backtests
             await asyncio.sleep(1)
             trader.process_prediction(prediction, latest_data=row, margin_fraction=0.20)
             
        trader.print_status()

    logging.info("Backtest Completed.")
    trader.print_status()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold AI Trading Bot - Backtesting Engine")
    parser.add_argument("--symbol", type=str, default="xauusdt", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g., 1h, 4h)")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent candles to backtest (for AI mode)")
    parser.add_argument("--mode", type=str, default="quant", choices=["ai", "quant"], help="Backtest mode: 'ai' (slow, LLM-based) or 'quant' (fast, vectorized math)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_backtest(symbol=args.symbol, timeframe=args.timeframe, limit=args.limit, mode=args.mode))
    except KeyboardInterrupt:
        logging.info("Backtesting interrupted.")
import argparse
import asyncio
import logging
import uvicorn
from data_feed import BinanceFeed
from indicators import calculate_indicators
from ai_agent import TradingAIAgent
from paper_trader import PaperTrader
from app_state import state
from api import app
from sentiment import get_market_sentiment
from macro_data import get_macro_data
from regime_detector import get_current_regime
from rl_env import AssetTradingEnv

# --- LOGGING SETUP ---
# Basic config handles console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class MemoryHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        state.logs.append(log_entry)

# Add memory handler to root logger to capture all logs for the UI
root_logger = logging.getLogger()
memory_handler = MemoryHandler()
memory_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
root_logger.addHandler(memory_handler)

# Suppress uvicorn access logs to keep the console clean
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# ---------------------

async def analysis_loop(feed_1h, feed_4h, feed_1d, agent, trader):
    await asyncio.sleep(3)  # Brief wait for initial WS connection & data
    while True:
        # Fetch dynamic settings
        interval_seconds = state.settings.get("ai_interval", 45)
        auto_trade = state.settings.get("auto_trade", True)
        trade_margin = state.settings.get("trade_margin", 0.20)
        
        try:
            df_1h = await feed_1h.get_dataframe()
            df_4h = await feed_4h.get_dataframe()
            df_1d = await feed_1d.get_dataframe()
            
            if df_1h.empty:
                logging.warning("1h Dataframe is empty, skipping analysis.")
                await asyncio.sleep(interval_seconds)
                continue
                
            logging.info("Calculating technical indicators...")
            df_with_indicators_1h = calculate_indicators(df_1h)
            df_with_indicators_4h = calculate_indicators(df_4h) if not df_4h.empty else None
            df_with_indicators_1d = calculate_indicators(df_1d) if not df_1d.empty else None
            
            # Ensure indicators exist before dropping NA for 1H
            required_cols = ['ema_200', 'rsi', 'macd', 'adx', 'atr']
            if not all(col in df_with_indicators_1h.columns for col in required_cols):
                logging.info("Not enough data to calculate all indicators yet.")
                await asyncio.sleep(interval_seconds)
                continue

            valid_data_1h = df_with_indicators_1h.dropna(subset=required_cols)
            if valid_data_1h.empty:
                logging.info("Not enough data to calculate all indicators yet.")
                await asyncio.sleep(interval_seconds)
                continue

            latest_1h = valid_data_1h.iloc[-1]
            latest_4h = df_with_indicators_4h.iloc[-1] if df_with_indicators_4h is not None and not df_with_indicators_4h.empty else None
            latest_1d = df_with_indicators_1d.iloc[-1] if df_with_indicators_1d is not None and not df_with_indicators_1d.empty else None

            trader.update_atr(latest_1h.get('atr', 0))
            logging.info(f"Triggering AI prediction for {feed_1h.symbol.upper()} at {latest_1h['close']}")
            
            # Fetch sentiment and macro data
            sentiment_data = get_market_sentiment()
            macro_data = await asyncio.to_thread(get_macro_data)
            
            # Fetch real-time Level 2 Order Book Imbalance
            order_book = await feed_1h.get_order_book()
            latest_1h['order_book_imbalance'] = order_book.get('imbalance', 0)
            
            state.current_sentiment = sentiment_data # Save for UI if needed
            state.quant_metrics = {
                "vwap": latest_1h.get("vwap", 0),
                "z_score": latest_1h.get("z_score", 0),
                "vol_imbalance": latest_1h.get("vol_imbalance_ema", 0),
                "volatility_regime": latest_1h.get("volatility_regime", 0),
                "ob_imbalance": order_book.get('imbalance', 0)
            }
            
            # Fetch recent trades to give the AI context on its past performance
            recent_trades = trader.get_recent_trades(limit=3)
            
            # The agent.analyze method returns the full JSON prediction from Gemini
            # We also want to extract what the local RL agent thought just before that
            # Since rl_action is calculated inside analyze, we can modify agent.analyze to return a tuple, 
            # OR we can just calculate the rl_action here in main.py for the UI.
            # Let's calculate it here quickly for the UI since the model is loaded in the agent:
            
            # V4: compute regime for UI display
            try:
                import numpy as np
                regime_data = get_current_regime(latest_1h, hist_df=valid_data_1h)
                state.regime = regime_data
            except Exception as e:
                logging.error(f"Regime detection error: {e}")
                state.regime = {"regime": "UNKNOWN"}

            # V4: RL inference uses full 15-feature observation
            rl_action = "NEUTRAL"
            if agent.rl_model:
                try:
                    import numpy as np
                    features  = AssetTradingEnv.FEATURE_COLUMNS
                    obs_vals  = [float(latest_1h.get(col, 0.0)) for col in features]
                    obs_vals.extend([0.0, 0.0, 0.0])  # position=0, current_risk_fraction=0, unrealised_pnl=0
                    obs = np.nan_to_num(np.array(obs_vals, dtype=np.float32), nan=0.0)
                    action, _ = agent.rl_model.predict(obs, deterministic=True)
                    
                    # Continuous Action Space decoding
                    raw_conviction = float(np.clip(action[0], -1.0, 1.0)) if isinstance(action, (np.ndarray, list)) else float(action)
                    conviction = round(raw_conviction * 10) / 10.0
                    if abs(conviction) < 0.1:
                        rl_action = "NEUTRAL"
                    elif conviction > 0:
                        rl_action = "LONG"
                    else:
                        rl_action = "SHORT"
                except Exception as e:
                    logging.error(f"UI RL Prediction error: {e}")

            state.rl_prediction = rl_action

            prediction = await agent.analyze(
                latest_1h, latest_4h, latest_1d, recent_trades, sentiment_data, macro_data,
                hist_df=valid_data_1h  # V4: pass history for regime percentile context
            )
            logging.info(f"AI Prediction Response:\n{prediction}")
            
            state.latest_prediction = prediction # Update global state for UI
            
            if auto_trade:
                trader.process_prediction(prediction, latest_data=latest_1h, margin_fraction=trade_margin)
            else:
                logging.info("Auto-trade is disabled in settings. Skipping execution.")
                
            trader.print_status()
            
        except Exception as e:
            logging.error(f"Error in analysis loop: {e}")
            
        await asyncio.sleep(interval_seconds)

async def price_monitor_loop(feed, trader, interval_seconds=1):
    await asyncio.sleep(3) # Wait for initial WS connection
    while True:
        try:
            df = await feed.get_dataframe()
            if not df.empty:
                current_price = df.iloc[-1]['close']
                state.current_price = current_price # Update global state for UI
                trader.update_price(current_price)
        except Exception as e:
            pass 
        await asyncio.sleep(interval_seconds)

async def start_api():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    logging.info("🌐 Web UI available at: http://127.0.0.1:8000")
    await server.serve()

async def main(use_ui=True):
    logging.info(f"Starting {args.symbol.upper()} Price Prediction & Trading System...")
    agent = TradingAIAgent(symbol=args.symbol)
    
    trader = PaperTrader(initial_balance=10000.0, symbol=args.symbol)
    state.trader = trader # Link to global state for UI
    
    feed_1h = BinanceFeed(symbol=args.symbol, timeframe="1h")
    feed_4h = BinanceFeed(symbol=args.symbol, timeframe="4h")
    feed_1d = BinanceFeed(symbol=args.symbol, timeframe="1d")
    
    await asyncio.gather(feed_1h.initialize(), feed_4h.initialize(), feed_1d.initialize())
    
    tasks = [
        asyncio.create_task(feed_1h.start_feed()),
        asyncio.create_task(feed_1h.start_depth_feed()), # Level 2 Order book
        asyncio.create_task(feed_4h.start_feed()),
        asyncio.create_task(feed_1d.start_feed()),
        asyncio.create_task(analysis_loop(feed_1h, feed_4h, feed_1d, agent, trader)),
        asyncio.create_task(price_monitor_loop(feed_1h, trader, interval_seconds=1))
    ]
    
    if use_ui:
        tasks.append(asyncio.create_task(start_api()))
        
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold AI Trading Bot")
    parser.add_argument("--symbol", type=str, default="xauusdt", help="Asset symbol to trade (e.g., xauusdt, btcusdt)")
    parser.add_argument("--no-ui", action="store_true", help="Run without the Web Dashboard (CLI mode only)")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(use_ui=not args.no_ui))
    except KeyboardInterrupt:
        logging.info("System shutting down.")

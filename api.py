from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app_state import state
import os
import json
import logging

app = FastAPI()

# Ensure static dir exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class TradeRequest(BaseModel):
    direction: str
    target: float
    stop_loss: float
    margin: float = 0.20

class SettingsRequest(BaseModel):
    auto_trade: bool
    trade_margin: float
    ai_interval: int
    discord_webhook: str = ""
    dynamic_sizing: bool = True

@app.on_event("startup")
def load_db_settings():
    db_settings = db.get_settings()
    for k, v in db_settings.items():
        state.settings[k] = v
    logging.info(f"Loaded settings from DB: {state.settings}")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/settings")
def get_settings():
    return state.settings

@app.post("/api/settings")
def update_settings(new_settings: SettingsRequest):
    state.settings["auto_trade"] = new_settings.auto_trade
    state.settings["trade_margin"] = new_settings.trade_margin
    state.settings["ai_interval"] = new_settings.ai_interval
    state.settings["discord_webhook"] = new_settings.discord_webhook
    state.settings["dynamic_sizing"] = new_settings.dynamic_sizing
    
    # Save to database
    for k, v in state.settings.items():
        db.save_setting(k, v)
        
    logging.info(f"⚙️ Settings saved to DB: Auto Trade={new_settings.auto_trade}, Dynamic Sizing={new_settings.dynamic_sizing}")
    return {"status": "success"}

from database import db

@app.get("/api/data")
def get_data():
    trader_data = state.trader.get_state() if state.trader else {}
    
    # Parse prediction string to dict if needed for cleaner JSON response
    prediction_dict = state.latest_prediction
    if isinstance(state.latest_prediction, str):
        try:
            prediction_dict = json.loads(state.latest_prediction)
        except Exception:
            pass

    sentiment_data = getattr(state, 'current_sentiment', {"value": 50, "classification": "Neutral"})

    return {
        "price": state.current_price,
        "prediction": prediction_dict,
        "rl_prediction": getattr(state, 'rl_prediction', 'NEUTRAL'),
        "trader": trader_data,
        "logs": list(state.logs),
        "settings": state.settings,
        "portfolio_history": db.get_portfolio_history(),
        "sentiment": sentiment_data,
        "quant_metrics": state.quant_metrics,
        "regime": getattr(state, "regime", {}),
    }

@app.get("/api/backtest")
def get_backtest_results():
    try:
        if os.path.exists("backtest_results.json"):
            with open("backtest_results.json", "r") as f:
                return json.load(f)
        return {"error": "No backtest results available."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/markets")
def get_markets_data():
    from macro_data import get_macro_data
    return get_macro_data()

@app.get("/api/regime")
def get_regime():
    """V4: Returns current market regime + confluence metadata."""
    regime = getattr(state, 'regime', {})
    prediction = state.latest_prediction
    ensemble = {}
    if isinstance(prediction, str):
        try:
            import json as _json
            p = _json.loads(prediction)
            ensemble = p.get('_ensemble', {})
        except Exception:
            pass
    return {"regime": regime, "ensemble": ensemble}

@app.post("/api/run_backtest")
def run_backtest_endpoint():
    import subprocess
    try:
        # Run it non-blocking
        subprocess.Popen(["python", "backtest.py", "--mode", "quant"])
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/optimize")
def run_optimization_endpoint():
    import subprocess
    try:
        # Run it non-blocking
        subprocess.Popen(["python", "optimize.py"])
        return {"status": "started"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/trade")
def place_trade(trade: TradeRequest):
    if not state.trader:
        return {"error": "Trader not initialized"}
    
    if state.current_price == 0.0:
        return {"error": "Price data not yet available. Please wait."}
        
    try:
        position_type = "LONG" if trade.direction.upper() == "LONG" else "SHORT"
        
        # Validations
        if position_type == "LONG":
            if trade.target <= state.current_price or trade.stop_loss >= state.current_price:
                return {"error": f"Invalid LONG: TP must be > {state.current_price} and SL must be < {state.current_price}"}
        else:
            if trade.target >= state.current_price or trade.stop_loss <= state.current_price:
                return {"error": f"Invalid SHORT: TP must be < {state.current_price} and SL must be > {state.current_price}"}

        state.trader._open_position(position_type, trade.target, trade.stop_loss, "Manual Trade triggered via UI Dashboard", trade.margin)
        logging.info(f"👨‍💻 Manual {position_type} order placed from UI with {trade.margin*100}% margin.")
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Manual trade error: {e}")
        return {"error": str(e)}

class CloseRequest(BaseModel):
    position_index: int

@app.post("/api/close_position")
def close_active_position(req: CloseRequest):
    if not state.trader:
        return {"error": "Trader not initialized"}
        
    if not state.trader.positions or req.position_index >= len(state.trader.positions):
        return {"error": "Invalid position index."}
        
    try:
        pos = state.trader.positions[req.position_index]
        state.trader._close_position(pos, "MANUAL_CLOSE", state.current_price)
        logging.info(f"👨‍💻 Manually closed position at index {req.position_index} from UI.")
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Manual close error: {e}")
        return {"error": str(e)}

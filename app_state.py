import collections

class AppState:
    def __init__(self):
        self.current_price = 0.0
        self.latest_prediction = None
        self.quant_metrics = {}
        self.logs = collections.deque(maxlen=200)
        self.trader = None
        self.rl_prediction = "NEUTRAL"
        self.current_sentiment = None
        # V4: regime detection state (populated each analysis cycle)
        self.regime = {
            "regime": "UNKNOWN",
            "adx": 0,
            "volatility_percentile": 50,
            "confidence": 0,
            "signal_bias": "MEAN_REVERSION",
            "preferred_strategy": "Awaiting first analysis cycle..."
        }
        self.settings = {
            "auto_trade": True,
            "trade_margin": 0.20,
            "ai_interval": 45,
            "discord_webhook": "",
            "dynamic_sizing": True
        }

state = AppState()

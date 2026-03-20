import os
import re

app_dir = 'E:/AI/testing_cryp/app_claude'

def update_rl_env():
    path = os.path.join(app_dir, 'rl_env.py')
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    
    # Replace absolute features with relative ones
    content = content.replace(\"'close', 'rsi', 'macd'\", \"'return_1', 'rsi', 'macd'\")
    content = content.replace(\"'vwap'\", \"'return_5'\")
    
    # Generalize class and logs
    content = content.replace(\"GoldTradingEnv\", \"AssetTradingEnv\")
    content = content.replace(\"rl_env.py â€” GoldTradingEnv\", \"rl_env.py â€” AssetTradingEnv\")
    
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    print(\"Updated rl_env.py\")

def update_ai_agent():
    path = os.path.join(app_dir, 'ai_agent.py')
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    
    # Rename class
    content = content.replace(\"class GoldAIAgent:\", \"class TradingAIAgent:\")
    content = content.replace(\"Analyze XAU/USDT\", \"Analyze {self.symbol.upper()}\")
    content = content.replace(\"expert gold market analyst\", \"expert financial market analyst\")
    content = content.replace(\"GoldTradingEnv\", \"AssetTradingEnv\")
    
    # Update __init__ to accept symbol and dynamic model path
    init_old = \"def __init__(self):\"
    init_new = \"def __init__(self, symbol='xauusdt'):\"
    content = content.replace(init_old, init_new)
    
    content = content.replace(\"self.rl_model = None\", \"self.rl_model = None\n        self.symbol = symbol\")
    
    # Dynamic model path loading logic
    content = re.sub(r'model_path = \"models/ppo_gold_real_final\.zip\"', 
                     'model_path = f\"models/ppo_{self.symbol}_final.zip\"', content)
    
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    print(\"Updated ai_agent.py\")

def update_main():
    path = os.path.join(app_dir, 'main.py')
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    
    # Import updates
    content = content.replace(\"from ai_agent import GoldAIAgent\", \"from ai_agent import TradingAIAgent\")
    content = content.replace(\"from rl_env import GoldTradingEnv\", \"from rl_env import AssetTradingEnv\")
    
    # Argparse updates
    content = content.replace('description=\"Gold AI Trading Bot\"', 'description=\"Asset AI Trading Bot\"')
    # Use a more specific replace for argparse
    arg_old = 'parser.add_argument(\"--no-ui\", action=\"store_true\", help=\"Run without the Web Dashboard (CLI mode only)\")'
    arg_new = 'parser.add_argument(\"--symbol\", type=str, default=\"xauusdt\", help=\"Asset symbol to trade (e.g., xauusdt, btcusdt)\")\n    ' + arg_old
    content = content.replace(arg_old, arg_new)
    
    # main() logic updates
    content = content.replace(\"agent = GoldAIAgent()\", \"agent = TradingAIAgent(symbol=args.symbol)\")
    content = content.replace(\"feed_1h = BinanceFeed(symbol=\\\"xauusdt\\\", timeframe=\\\"1h\\\")\", \"feed_1h = BinanceFeed(symbol=args.symbol, timeframe=\\\"1h\\\")\")
    content = content.replace(\"feed_4h = BinanceFeed(symbol=\\\"xauusdt\\\", timeframe=\\\"4h\\\")\", \"feed_4h = BinanceFeed(symbol=args.symbol, timeframe=\\\"4h\\\")\")
    content = content.replace(\"feed_1d = BinanceFeed(symbol=\\\"xauusdt\\\", timeframe=\\\"1d\\\")\", \"feed_1d = BinanceFeed(symbol=args.symbol, timeframe=\\\"1d\\\")\")
    
    content = content.replace(\"GoldTradingEnv.FEATURE_COLUMNS\", \"AssetTradingEnv.FEATURE_COLUMNS\")
    content = content.replace('Starting Gold Price Prediction', 'f\"Starting {args.symbol.upper()} Price Prediction\"')
    
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    print(\"Updated main.py\")

if __name__ == \"__main__\":
    update_rl_env()
    update_ai_agent()
    update_main()

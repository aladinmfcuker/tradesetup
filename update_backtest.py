import os
import json
import numpy as np
import pandas as pd

path = 'E:/AI/testing_cryp/app_claude/backtest.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

# Replace GoldAIAgent with TradingAIAgent
content = content.replace('GoldAIAgent', 'TradingAIAgent')

# Update run_vectorized_quant_backtest to include Sortino and Max DD Duration
old_results_block = \"\"\"    results = {
        \"total_candles\": len(df),
        \"total_trades\": int(total_trades),
        \"win_rate\": float(win_rate),
        \"market_return\": float(final_market),
        \"strategy_return\": float(final_strat)
    }\"\"\"

new_results_block = \"\"\"
    # Calculate Drawdown
    cum_rets = df['cum_strategy_return']
    running_max = cum_rets.cummax()
    drawdown = (cum_rets - running_max) / running_max
    max_drawdown = drawdown.min()

    # Drawdown Duration
    is_in_drawdown = drawdown < 0
    drawdown_counts = is_in_drawdown.groupby((~is_in_drawdown).cumsum()).cumcount()
    max_dd_duration = drawdown_counts.max()

    # Sortino Ratio (Downside deviation only)
    strat_rets = df['strategy_return'].dropna()
    downside_rets = strat_rets[strat_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252 * 24)
    sortino = (strat_rets.mean() * 252 * 24) / downside_std if downside_std > 0 else 0

    results = {
        \"total_candles\": len(df),
        \"total_trades\": int(total_trades),
        \"win_rate\": float(win_rate),
        \"market_return\": float(final_market),
        \"strategy_return\": float(final_strat),
        \"max_drawdown_pct\": float(max_drawdown * 100),
        \"max_drawdown_duration_bars\": int(max_dd_duration),
        \"sortino_ratio\": float(sortino)
    }
\"\"\"

content = content.replace(old_results_block, new_results_block)

with open(path, 'w', encoding='utf-8') as f: f.write(content)
print('Updated backtest.py with institutional metrics')

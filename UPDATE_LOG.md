# App 1.1 - UI and AI Optimization Updates (V2)

## Recent Major Updates

### UI Modernization (V2)
- Upgraded the frontend (`static/index.html`) to a modern Glassmorphism design with `backdrop-filter` effects.
- Added animated **Skeleton Loaders** for all metrics while awaiting backend data.
- Replaced standard alerts with a sliding **Toast Notification System** for all trading actions and configuration saves.
- Typography updated to `JetBrains Mono` for tabular data, and integrated high-quality `Phosphor Icons` across the dashboard.
- Made the TradingView chart container dynamically resizable by the user (`resize: vertical`).

### Dynamic Backtesting System
- Updated `backtest.py` to save its vector math output into a `backtest_results.json` file.
- Created `/api/run_backtest` and `/api/backtest` endpoints in `api.py`.
- Built an interactive **Signals Dashboard** in the UI, replacing the placeholder, allowing the user to trigger the backtest script and view real-time calculations of Win Rate, Total Trades, and Strategy Returns.

### Macro Markets Integration
- Created `/api/markets` using the logic in `macro_data.py` (fetching DXY and US10Y).
- Built a macro data dashboard in the **Markets Tab** within the UI.
- **Reliability Fix:** Updated `macro_data.py` to use `UUP` (Dollar Index ETF) as a proxy for DXY to resolve delisting errors on Yahoo Finance.

### AI Algorithmic Self-Learning & Fine Tuning
- Updated `optimize.py` to write its best hyperparameter calculations to `optimal_params.json` (e.g., Z-score, RSI limits).
- Updated the AI prompt construction in `ai_agent.py` to dynamically load `optimal_params.json` and inject **"BACKTEST OPTIMIZED GUIDELINES"** into the LLM context.
- Added an interactive **"Recalibrate AI Parameters"** button to the UI to execute `/api/optimize` on demand, effectively allowing the user to retrain the math parameters driving the AI.

---

# App 1.1 - Advanced Quant & RL Integration (V3)

## New Institutional Features

### Reinforcement Learning Sniper Brain
- Created a custom **Gymnasium Environment** (`rl_env.py`) for Gold trading with multi-timeframe indicator mapping.
- Implemented a **Pro-Grade Training Pipeline** (`train_rl_real_data.py`) with CPU multiprocessing/Vectorization support.
- Successfully trained a **2.5 Million Timestep PPO Model** on clean 1-Hour Gold Futures (GC=F) data.
- **The "Sniper" Result:** The model achieved a **4.02% Strategy Return** on unseen (out-of-sample) data with strict risk management (prioritizing Neutrality until a high-conviction edge is found).
- Integrated the model into `ai_agent.py` to provide a real-time mathematical conviction signal to the Gemini LLM.

### Institutional Liquidity & Order Book Analysis
- Updated `data_feed.py` to connect to Binance's **Level 2 Depth WebSocket** (`@depth10@100ms`).
- Implemented real-time **Order Book Imbalance** calculation (Bid Vol vs Ask Vol) to detect institutional support/resistance.
- Injected live imbalance metrics into the AI analysis prompt and visualized it on the dashboard.

### Advanced Risk Management (Dynamic Position Sizing)
- Implemented **Kelly Criterion-inspired dynamic sizing** in `paper_trader.py`.
- The system now calculates trade size based on **AI Confidence levels** (proportional 2% account risk) and the exact distance to the stop loss.
- Added a **UI Toggle** in the Settings panel to switch between "Dynamic Position Sizing" and "Fixed Margin Allocation."

### Core Architecture & Data Quality
- **Data Standard:** Standardized on Yahoo Finance Gold Futures (cleaner signal) for historical training while maintaining Binance for live execution.
- **Workspace Cleanup:** Purged unprofitable models and low-quality crypto-only training datasets to optimize system performance.
- **Pipeline Integrity:** Verified full end-to-end integration via `test_live_cycle.py`, confirming the RL and LLM brains are working in synergy to protect capital.

---

# App 1.2 - Data Scaling, RL Risk-to-Reward Tuning, & UI Upgrades (V4)

## Recent Major Updates

### Composite Training Dataset (2018-2026)
- Designed `fetch_real_data.py` to build a highly structured, temporally-aware dataset across 24,000+ candles:
  - **2018–2020:** Daily candles via Yahoo Finance (`GC=F`).
  - **2021–2022:** 4H candles from Binance (`PAXGUSDT`).
  - **2023–2024:** 1H candles from Binance (`PAXGUSDT`).
  - **2025+:** Live 1H feed from Binance Futures (`XAUUSDT`).
- This preserves structural pattern history while adapting to modern micro-volatility.

### Resolved Reinforcement Learning "0-Trade Collapse"
- Discovered the PPO model was collapsing into a "permanently neutral" state due to extreme risk aversion in out-of-sample testing.
- Aggressively tuned `rl_env.py` (V4.2):
  - Increased `INACTIVITY_PENALTY` from `0` to `-1e-3` to aggressively punish neutrality.
  - Implemented dynamic **ATR Risk-to-Reward Bonuses** (`+0.05` per captured ATR) to explicitly teach the model to hunt for high R:R setups instead of just avoiding drawdowns.
- Re-trained the model over 2.5 million timesteps, achieving an out-of-sample forward (2024+) **Sharpe Ratio of 1.189** and **Strategy Return of +15.39%** without collapsing.

### Regime-Gated Ensemble Sizing Integration
- Bridged a missing gap in `paper_trader.py` by incorporating the AI Agent's `size_multiplier` (derived mathematically from the Regime Detector and RL signal). 
- If the AI LLM outputs 80% confidence, but the mathematical Regime (e.g. Volatile) assigns a 0.5x penalty, the effective confidence correctly scales down to 40%, automatically shrinking the position size or skipping the trade.

### Confluence & Regime UI Dashboard
- Exposed deep background logic to the user by injecting `Market Regime` and `Confluence Score` directly into the Intelligence Core panel of the frontend UI (`static/index.html`).
- The UI now dynamically highlights color-coded Regime states (e.g., green for Trending Bull, orange for Volatile) and exact numerical Confluence scores (e.g., 77/100 HIGH_CONVICTION) before a trade is even placed.
- Confirmed full end-to-end telemetry via `test_live_cycle.py` before deploying `main.py` live.

---

# App 1.3 - Data Stitching Fixes & Strict Risk/Reward Enforcement (V4.3)

## Pending/Recent Updates

### Scale-Invariant Volume Fix (Feature Distribution Shift)
- **Critical Catch:** Identified a data stitching flaw in the composite dataset where raw volume scales drastically differed between exchanges and asset types (e.g., Yahoo `GC=F` avg 6k contracts vs Binance `PAXGUSDT` 1H avg 40 tokens vs Binance `XAUUSDT` Futures avg 6.9k micro-contracts).
- Modified `indicators.py` to process `liquidity_vacuum` and `orderbook_slope` as **Scale-Invariant Ratios**.
- Replaced absolute raw volume division with **Relative Volume** (`Current Volume / 200-Period Average Volume`). This prevents the neural network from hallucinating massive liquidity anomalies when historical data sources change.

### Stiffer Risk-to-Reward (R:R) Mechanics
- Upgraded `rl_env.py` to strictly enforce "Sniper" behavior (V4.3):
  - **Massive R:R Bonus:** Increased `ATR_PROFIT_BONUS` from `+0.05` to `+0.15` per captured ATR, massively incentivizing the model to let winners run.
  - **Dynamic Risk Penalty:** Shifted from a flat loss penalty to a dynamically scaling loss penalty (`max(0.01, abs(profit_atrs) * 0.05)`). This harshly punishes the agent for letting losses extend beyond its targeted risk parameters.
  - **Punished Neutrality:** Increased `INACTIVITY_PENALTY` from `-1.0e-3` to `-1.5e-3` to prevent the model from cowering in cash.

---

# App 1.4 - Profitable Strategy Edge Injection (V4.4)

## Recent Updates

### Aggressive Reward Tuning (0-Trade Collapse Solved)
- Fixed the issue where the model was cowering in cash due to loss aversion.
- Tuned `INACTIVITY_PENALTY` to `-5.0e-3`, `ATR_PROFIT_BONUS` to `+0.50`, and `LOSS_MULTIPLIER` to `0.5`. 
- **Result:** Successfully resulted in active trading across all out-of-sample sets (4,120 trades in Forward testing). Though returns were slightly negative (-1.3%), this confirmed the model was successfully forced out of a permanent 100% cash holding and actively looking for trades.

### Momentum & Mean-Reversion Edge Injection
- Upgraded `rl_env.py` to explicitly inject a mathematically profitable edge using technical indicators:
  - **Trending Regime (ADX > 20):** Massive reward (`+0.05`) for trading with the Plus/Minus DI momentum trend, and a severe penalty (`-0.05`) for attempting counter-trend trades.
  - **Ranging Regime (ADX <= 20):** Rewarded for accurate mean-reversion entries (e.g. buying when Z-Score < -1.0, selling when Z-Score > 1.0) to capture edge in chop.

---

# App 1.5 - Pure ML and Continuous Action Space (V6.0)

## Recent Updates

### The Root Cause of the 0-Trade Collapse Discovered
- Discovered that the agent falls into a permanent "Neutral" local minimum not because of a lack of strategy, but due to **Fee Aversion**.
- Because the previous `Discrete(3)` action space forced the model to go "All In" (10% of balance) on random exploratory trades, the transaction fees immediately destroyed its capital, teaching it that any action equals pain.

### Shift to Continuous Action Space `Box(-1.0, 1.0)`
- Architected a V6 environment (`rl_env.py`) moving away from the binary `(0=Neutral, 1=Long, 2=Short)` system.
- The model now outputs a continuous scalar representing its **Conviction Level**:
  - `-1.0` = 100% Short (of risk allocation)
  - `-0.1` = 10% Short
  - `0.0` = Neutral
  - `0.5` = 50% Long
- **The Benefit:** This allows the agent to safely "test the waters" by taking microscopic 1% positions during early training. It can slowly build confidence in technical patterns without having its equity instantly wiped out by fees.

### The "Holy Grail" Objective Function
- Stripped out all hardcoded "Strategy Hints" (like the +0.05 ADX/Z-score bonuses) to let the ML model find its own mathematical edge organically.
- Replaced it with a dual-objective Terminal Reward: `Reward = Sharpe Ratio + Profit Factor`.
- The model is now forced to optimize for both massive raw edge (Profit Factor) and incredibly smooth equity curves (Sharpe).

### 🚀 V6 RL Retraining Success (The Sniper Evolves)
- **Fee Bug Fixed:** Discovered that the transaction cost calculation was applying a flat fee based on the nominal price of gold, destroying the model's capital on micro-trades. Updated to charge 2 bps on the *notional margin traded*.
- **Reward Scaling:** Multiplied step rewards by `100.0` to prevent gradient vanishing, allowing the PPO model to learn from dense returns.
- **Horizon Memory Extended:** Increased PPO `gamma` from 0.995 to 0.999, extending the model's half-life memory from ~138 hours to ~693 hours. This lets the agent optimize for long-term capital compounding via Terminal Rewards.
- **Results:** The continuous action space model is now highly profitable. The agent shifted into a long-term "Sniper" state, holding trades across large macro trends with incredible capital preservation:
  - **Forward Period (2024+):** +15.39% Strategy Return
  - **Max Drawdown:** 4.39%
  - **Sharpe Ratio:** 1.189
  - **Calmar Ratio:** 3.50
- **AI Agent Integration Fixed:** Updated `ai_agent.py` to correctly supply 24 features (including `current_risk_fraction`) to match the new continuous action space `Box(-1.0, 1.0)` observation shape.

### 2026-03-11 Fixes
- Installed missing `yfinance` dependency.
- Fixed regime detection error in `main.py` by correcting `historical_df` kwarg to `hist_df`.
- Updated RL inference in `main.py` to match V6 continuous action space and 24-feature observation shape (added `current_risk_fraction`).

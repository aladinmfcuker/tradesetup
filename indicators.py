"""
indicators.py  —  Technical Indicator Pipeline (V4 Enhanced)

New in V4:
  - plus_di / minus_di  : raw directional indicators (required by RegimeDetector)
  - vwap_dev_pct        : % deviation from VWAP (mean-reversion signal quality)
  - bb_width            : Bollinger Band width as % of middle (squeeze detector)
  - realized_vol_20     : 20-bar rolling realised volatility (annualised %)
  - session_score       : continuous float scoring time-of-day proximity to
                          high-volume sessions (London 07-16 UTC, NY 13-22 UTC)
  - return_1            : single-bar log return (feature for RL env)
  - return_5            : 5-bar log return (momentum feature for RL env)
"""

import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return df

    df_ta = df.copy()

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta    = df_ta['close'].diff()
    up       = delta.clip(lower=0)
    down     = -1 * delta.clip(upper=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs       = ema_up / ema_down
    df_ta['rsi'] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = df_ta['close'].ewm(span=12, adjust=False).mean()
    ema26 = df_ta['close'].ewm(span=26, adjust=False).mean()
    df_ta['macd']        = ema12 - ema26
    df_ta['macd_signal'] = df_ta['macd'].ewm(span=9, adjust=False).mean()
    df_ta['macd_hist']   = df_ta['macd'] - df_ta['macd_signal']

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df_ta['bb_middle'] = df_ta['close'].rolling(window=20).mean()
    std                = df_ta['close'].rolling(window=20).std()
    df_ta['bb_upper']  = df_ta['bb_middle'] + 2 * std
    df_ta['bb_lower']  = df_ta['bb_middle'] - 2 * std
    # BB Width % — squeeze when width narrows (breakout precursor)
    df_ta['bb_width']  = (df_ta['bb_upper'] - df_ta['bb_lower']) / df_ta['bb_middle'].replace(0, np.nan) * 100

    # ── EMAs ─────────────────────────────────────────────────────────────────
    df_ta['ema_20']  = df_ta['close'].ewm(span=20,  adjust=False).mean()
    df_ta['ema_50']  = df_ta['close'].ewm(span=50,  adjust=False).mean()
    df_ta['ema_200'] = df_ta['close'].ewm(span=200, adjust=False).mean()

    # ── ATR ──────────────────────────────────────────────────────────────────
    high_low    = df_ta['high'] - df_ta['low']
    high_close  = np.abs(df_ta['high'] - df_ta['close'].shift())
    low_close   = np.abs(df_ta['low']  - df_ta['close'].shift())
    true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_ta['atr'] = true_range.rolling(14).mean()

    # ── ADX + Plus/Minus DI (V4: expose raw DI for regime detector) ──────────
    plus_dm  = df_ta['high'].diff().clip(lower=0)
    minus_dm = (-df_ta['low'].diff()).clip(lower=0)
    # DM exclusion rule
    plus_dm[plus_dm <= minus_dm]  = 0
    minus_dm[minus_dm <= plus_dm] = 0

    tr14          = true_range.rolling(14).sum()
    df_ta['plus_di']  = 100 * (plus_dm.rolling(14).sum()  / tr14.replace(0, np.nan))
    df_ta['minus_di'] = 100 * (minus_dm.rolling(14).sum() / tr14.replace(0, np.nan))
    dx            = 100 * np.abs(df_ta['plus_di'] - df_ta['minus_di']) / \
                    (df_ta['plus_di'] + df_ta['minus_di']).replace(0, np.nan)
    df_ta['adx']  = dx.rolling(14).mean()

    # ── Fibonacci Retracements (Rolling 50-period) ───────────────────────────
    rolling_high = df_ta['high'].rolling(50).max()
    rolling_low  = df_ta['low'].rolling(50).min()
    diff = rolling_high - rolling_low
    df_ta['fib_0.382'] = rolling_high - diff * 0.382
    df_ta['fib_0.500'] = rolling_high - diff * 0.500
    df_ta['fib_0.618'] = rolling_high - diff * 0.618

    # ── VWAP (rolling 24-period) ─────────────────────────────────────────────
    typical_price = (df_ta['high'] + df_ta['low'] + df_ta['close']) / 3
    vol_safe      = df_ta['volume'].replace(0, np.nan)
    df_ta['vwap'] = (typical_price * vol_safe).rolling(window=24).sum() / \
                    vol_safe.rolling(window=24).sum()

    # VWAP Deviation % — how far price is from institutional VWAP anchor
    df_ta['vwap_dev_pct'] = (df_ta['close'] - df_ta['vwap']) / df_ta['vwap'].replace(0, np.nan) * 100

    # ── Z-Score (Price vs EMA 50) ─────────────────────────────────────────────
    price_std        = df_ta['close'].rolling(window=50).std()
    df_ta['z_score'] = (df_ta['close'] - df_ta['ema_50']) / price_std.replace(0, np.nan)

    # ── Volume Imbalance ─────────────────────────────────────────────────────
    if 'taker_buy_base_asset_volume' in df_ta.columns:
        taker_sell               = df_ta['volume'] - df_ta['taker_buy_base_asset_volume']
        df_ta['vol_imbalance']   = (df_ta['taker_buy_base_asset_volume'] - taker_sell) / \
                                   vol_safe
        df_ta['vol_imbalance_ema'] = df_ta['vol_imbalance'].ewm(span=10, adjust=False).mean()
    else:
        df_ta['vol_imbalance']     = 0.0
        df_ta['vol_imbalance_ema'] = 0.0

    # ── Volatility Regime (ATR / Close %) ────────────────────────────────────
    df_ta['volatility_regime'] = (df_ta['atr'] / df_ta['close'].replace(0, np.nan)) * 100

    # ── Realised Volatility (20-bar, annualised) ─────────────────────────────
    log_ret = np.log(df_ta['close'] / df_ta['close'].shift(1))
    df_ta['realized_vol_20'] = log_ret.rolling(20).std() * np.sqrt(252 * 24) * 100  # 1H bars

    # ── Log Returns (momentum features for RL) ───────────────────────────────
    df_ta['return_1'] = log_ret
    df_ta['return_5'] = np.log(df_ta['close'] / df_ta['close'].shift(5))

    # ── Session Score (0-1, peaks at London/NY overlap 13-17 UTC) ───────────
    #   London: 07-16 UTC  |  New York: 13-22 UTC  |  Overlap: 13-16 UTC
    if 'timestamp' in df_ta.columns:
        try:
            ts       = pd.to_datetime(df_ta['timestamp'])
            hour_utc = ts.dt.hour + ts.dt.minute / 60.0

            def _session(h):
                london = max(0.0, 1.0 - abs(h - 11.5) / 4.5)   # peak 11:30
                newyork = max(0.0, 1.0 - abs(h - 17.5) / 4.5)  # peak 17:30
                overlap = max(0.0, 1.0 - abs(h - 14.0) / 1.5)  # peak 14:00
                return min(1.0, london * 0.4 + newyork * 0.3 + overlap * 0.3)

            df_ta['session_score'] = hour_utc.apply(_session)
        except Exception:
            df_ta['session_score'] = 0.5
    else:
        df_ta['session_score'] = 0.5

    # ── Microstructure Signals (Proxies for RL) ──────────────────────────────
    if 'taker_buy_base_asset_volume' in df_ta.columns:
        taker_buy = df_ta['taker_buy_base_asset_volume']
        taker_sell = df_ta['volume'] - taker_buy
        df_ta['bid_ask_pressure'] = taker_buy / df_ta['volume'].replace(0, np.nan)
        df_ta['trade_delta'] = taker_buy - taker_sell
    else:
        df_ta['bid_ask_pressure'] = 0.5
        df_ta['trade_delta'] = 0.0

    df_ta['spread_proxy'] = (df_ta['high'] - df_ta['low']) / df_ta['close'].replace(0, np.nan)
    df_ta['volume_spikes'] = df_ta['volume'] / df_ta['volume'].rolling(20).mean().replace(0, np.nan)    

    # Use relative volume to prevent scale disruption across different historical datasets
    vol_baseline = df_ta['volume'].rolling(200).mean().replace(0, np.nan)
    rel_vol = df_ta['volume'] / vol_baseline

    df_ta['liquidity_vacuum'] = (df_ta['high'] - df_ta['low']) / rel_vol.replace(0, np.nan)     
    # Orderbook slope proxy: price change per unit of relative volume
    df_ta['orderbook_slope'] = (df_ta['close'] - df_ta['open']) / rel_vol.replace(0, np.nan)    

    # Fill NaNs from new rolling operations
    df_ta[['volume_spikes', 'liquidity_vacuum', 'orderbook_slope']] = df_ta[['volume_spikes', 'liquidity_vacuum', 'orderbook_slope']].fillna(0)
    # ── Order Book Imbalance placeholder (replaced by live feed) ─────────────
    if 'order_book_imbalance' not in df_ta.columns:
        if 'ob_imbalance' in df_ta.columns:
            df_ta['order_book_imbalance'] = df_ta['ob_imbalance']
        else:
            np.random.seed(42)
            noise = np.random.normal(0, 0.1, len(df_ta))
            df_ta['order_book_imbalance'] = np.clip(
                np.sin(np.linspace(0, 20, len(df_ta))) * 0.5 + noise, -1, 1
            )

    return df_ta

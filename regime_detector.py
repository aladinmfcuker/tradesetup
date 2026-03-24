"""
regime_detector.py  —  Market Regime Detection + Confluence Scoring
---------------------------------------------------------------------
Classifies the current market into one of 4 regimes:
  TRENDING_BULL  — strong directional uptrend (ADX-confirmed)
  TRENDING_BEAR  — strong directional downtrend (ADX-confirmed)
  RANGING        — low ADX, mean-reverting conditions
  VOLATILE       — elevated ATR%, choppy, reduce risk

Also provides build_confluence_score() which combines:
  - RL signal alignment with the detected regime
  - Multi-timeframe RSI/MACD agreement
  - Regime conviction strength
  - Z-Score confirmation
into a single 0-100 conviction score that gates position sizing.
"""

import numpy as np
import pandas as pd
import logging

ADX_TREND_THRESHOLD    = 25
VOLATILITY_HIGH_PCT    = 80   # top 20% ATR/price = volatile
REGIME_TRENDING_BULL   = "TRENDING_BULL"
REGIME_TRENDING_BEAR   = "TRENDING_BEAR"
REGIME_RANGING         = "RANGING"
REGIME_VOLATILE        = "VOLATILE"


def classify_regime_series(df: pd.DataFrame, lookback: int = 100) -> pd.Series:
    """Vectorised regime labels for an entire DataFrame."""
    required = ['adx', 'volatility_regime']
    if any(c not in df.columns for c in required):
        return pd.Series(REGIME_RANGING, index=df.index)

    regimes  = pd.Series(REGIME_RANGING, index=df.index)
    adx      = df['adx'].fillna(0)
    vol_pct  = df['volatility_regime'].rolling(lookback, min_periods=20).rank(pct=True) * 100

    # Volatile regime
    is_volatile = vol_pct >= VOLATILITY_HIGH_PCT
    regimes[is_volatile] = REGIME_VOLATILE

    # Trending regimes
    is_trending = (~is_volatile) & (adx >= ADX_TREND_THRESHOLD)
    if 'ema_50' in df.columns:
        bull_bias = df['ema_50'].diff(5).fillna(0) > 0
    elif 'plus_di' in df.columns and 'minus_di' in df.columns:
        bull_bias = df['plus_di'] > df['minus_di']
    else:
        bull_bias = df['close'].diff(10).fillna(0) > 0

    regimes[is_trending &  bull_bias] = REGIME_TRENDING_BULL
    regimes[is_trending & ~bull_bias] = REGIME_TRENDING_BEAR

    # Ranging is the default (already set)
    return regimes


def get_current_regime(latest: dict, hist_df: pd.DataFrame = None) -> dict:
    """Single-bar regime dict with confidence + strategy hint."""
    adx        = float(latest.get('adx', 0) or 0)
    vol_regime = float(latest.get('volatility_regime', 0) or 0)
    ema_50     = float(latest.get('ema_50', 0) or 0)
    close      = float(latest.get('close', 0) or 0)
    plus_di    = float(latest.get('plus_di', 0) or 0)
    minus_di   = float(latest.get('minus_di', 0) or 0)

    vol_percentile = 50.0
    if hist_df is not None and 'volatility_regime' in hist_df.columns:
        h = hist_df['volatility_regime'].dropna()
        if len(h) > 10:
            vol_percentile = float((h <= vol_regime).mean() * 100)

    if vol_percentile >= VOLATILITY_HIGH_PCT:
        regime     = REGIME_VOLATILE
        confidence = min(100, int((vol_percentile - VOLATILITY_HIGH_PCT) * 5 + 50))
        signal_bias = "REDUCE_SIZE"
    elif adx >= ADX_TREND_THRESHOLD:
        if plus_di > 0 and minus_di > 0:
            bull = plus_di > minus_di
        elif ema_50 > 0 and close > 0:
            bull = close > ema_50
        else:
            bull = True
        regime     = REGIME_TRENDING_BULL if bull else REGIME_TRENDING_BEAR
        confidence = min(100, int((adx - ADX_TREND_THRESHOLD) * 2 + 50))
        signal_bias = "FOLLOW_TREND"
    else:
        regime     = REGIME_RANGING
        confidence = min(100, int((ADX_TREND_THRESHOLD - adx) * 2 + 40))
        signal_bias = "MEAN_REVERSION"

    _strategy = {
        REGIME_TRENDING_BULL: "MOMENTUM — buy EMA20 pullbacks, avoid shorts",
        REGIME_TRENDING_BEAR: "MOMENTUM — sell EMA20 rallies, avoid longs",
        REGIME_RANGING:       "MEAN_REVERSION — fade Z-Score extremes",
        REGIME_VOLATILE:      "DEFENSIVE — reduce size 50%, prefer NEUTRAL",
    }

    return {
        "regime": regime,
        "adx": round(adx, 2),
        "volatility_percentile": round(vol_percentile, 1),
        "confidence": confidence,
        "signal_bias": signal_bias,
        "preferred_strategy": _strategy.get(regime, "UNKNOWN"),
    }


def build_confluence_score(rl_action: str, regime_data: dict,
                           latest_1h: dict, latest_4h: dict = None,
                           latest_1d: dict = None) -> dict:
    """
    Composite conviction score (0-100) from 4 independent factors.

    Factor weights
    --------------
    RL signal × regime alignment   35 pts
    Multi-timeframe RSI/MACD        30 pts
    Regime conviction (ADX/vol)     20 pts
    Z-Score confirmation            15 pts
    """
    score      = 0
    components = {}
    regime     = regime_data.get("regime", REGIME_RANGING)

    # ── Factor 1: RL × Regime alignment ─────────────────────────────────────
    rl_pts    = 0
    rl_aligned = False
    z = float(latest_1h.get('z_score', 0) or 0)

    if rl_action == "LONG":
        if regime == REGIME_TRENDING_BULL:           rl_pts, rl_aligned = 35, True
        elif regime == REGIME_RANGING and z < -1.0:  rl_pts, rl_aligned = 25, True
        elif regime == REGIME_RANGING:               rl_pts = 12
        elif regime == REGIME_VOLATILE:              rl_pts = 5
        else:                                        rl_pts = -20 # Severe penalty for fighting the trend
    elif rl_action == "SHORT":
        if regime == REGIME_TRENDING_BEAR:           rl_pts, rl_aligned = 35, True
        elif regime == REGIME_RANGING and z > 1.0:   rl_pts, rl_aligned = 25, True
        elif regime == REGIME_RANGING:               rl_pts = 12
        elif regime == REGIME_VOLATILE:              rl_pts = 5
        else:                                        rl_pts = -20 # Severe penalty for fighting the trend
    else:  # NEUTRAL
        rl_pts, rl_aligned = 15, True   # staying flat is always safe

    score += rl_pts
    components['rl_regime_alignment'] = rl_pts

    # ── Factor 2: Multi-timeframe confluence ─────────────────────────────────
    tf_pts   = 0
    rsi_1h   = float(latest_1h.get('rsi', 50) or 50)
    macd_1h  = float(latest_1h.get('macd', 0) or 0)
    msig_1h  = float(latest_1h.get('macd_signal', 0) or 0)
    mbull_1h = macd_1h > msig_1h

    if latest_4h is not None:
        rsi_4h   = float(latest_4h.get('rsi', 50) or 50)
        macd_4h  = float(latest_4h.get('macd', 0) or 0)
        msig_4h  = float(latest_4h.get('macd_signal', 0) or 0)
        mbull_4h = macd_4h > msig_4h

        if rl_action == "LONG":
            tf_pts += 15 if (rsi_4h < 60 and mbull_4h) else 0
            tf_pts += 10 if (rsi_1h < 65 and mbull_1h) else 4
        elif rl_action == "SHORT":
            tf_pts += 15 if (rsi_4h > 40 and not mbull_4h) else 0
            tf_pts += 10 if (rsi_1h > 35 and not mbull_1h) else 4
        else:
            tf_pts += 10 if (40 < rsi_4h < 60) else 4
            tf_pts += 7  if (40 < rsi_1h < 60) else 3

        if latest_1d is not None:
            rsi_1d    = float(latest_1d.get('rsi', 50) or 50)
            ema50_1d  = float(latest_1d.get('ema_50', 0) or 0)
            ema200_1d = float(latest_1d.get('ema_200', 0) or 0)
            close     = float(latest_1h.get('close', 0) or 0)
            daily_bull = (rsi_1d < 65) and (ema50_1d > ema200_1d) and (close > ema200_1d)
            if (rl_action == "LONG" and daily_bull) or (rl_action == "SHORT" and not daily_bull):
                tf_pts += 5
    else:
        tf_pts = 12 if rl_action in ("LONG", "SHORT") else 8

    tf_pts = min(30, tf_pts)
    score += tf_pts
    components['multi_timeframe'] = tf_pts

    # ── Factor 3: Regime conviction ──────────────────────────────────────────
    reg_pts = max(0, min(20, int(regime_data.get('confidence', 50) * 0.20)))
    score  += reg_pts
    components['regime_conviction'] = reg_pts

    # ── Factor 4: Z-Score confirmation ───────────────────────────────────────
    z_pts = 0
    if regime == REGIME_RANGING:
        if rl_action == "LONG":
            z_pts = 15 if z < -1.75 else (10 if z < -1.25 else 3)
        elif rl_action == "SHORT":
            z_pts = 15 if z > 1.75  else (10 if z > 1.25  else 3)
        elif abs(z) < 0.5:
            z_pts = 8  # neutral is correct near the mean
    else:
        # Trending: mild pullback is ideal entry
        if rl_action == "LONG"  and -1.0 < z < 0.2: z_pts = 10
        elif rl_action == "SHORT" and -0.2 < z < 1.0: z_pts = 10

    score += z_pts
    components['z_score_confirmation'] = z_pts

    # ── Cap Score for Volatile Regimes ───────────────────────────────────────
    if regime == REGIME_VOLATILE and rl_action != "NEUTRAL":
        score = min(score, 54) # Force volatile trades to be WEAK at best, encouraging SKIP

    # ── Grade & recommendation ───────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= 75:
        grade = "HIGH_CONVICTION"
        rec   = "EXECUTE — full position size"
    elif score >= 55:
        grade = "MODERATE"
        rec   = "EXECUTE — reduced size (0.7x)"
    elif score >= 35:
        grade = "WEAK"
        rec   = "SKIP or minimal size (0.3x)"
    else:
        grade = "NO_EDGE"
        rec   = "STAY NEUTRAL — signals conflict"

    return {
        "total_score":            score,
        "grade":                  grade,
        "recommendation":         rec,
        "rl_aligned_with_regime": rl_aligned,
        "regime":                 regime,
        "components":             components,
    }

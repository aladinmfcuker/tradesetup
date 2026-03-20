"""
confluence.py — Multi-Timeframe Confluence Scoring Engine

Produces a single numeric conviction score (-10 to +10) and a
human-readable breakdown for the Gemini LLM prompt.

Score > +4  → HIGH BULLISH confluence  → full size allowed
Score < -4  → HIGH BEARISH confluence  → full size short allowed
|Score| < 2 → LOW confluence           → NEUTRAL / no trade recommended

Individual signal sources & weights:
  1H  EMA trend alignment       ±1.5
  4H  EMA trend alignment       ±2.0
  1D  EMA trend alignment       ±2.0
  1H  RSI bias                  ±1.0
  4H  RSI bias                  ±1.5
  1H  MACD direction            ±1.0
  1H  Z-Score mean-reversion    ±1.0  (inverse: extreme = signal)
  1H  ADX trend strength        ±0.0 (modifier only)
  Regime gate                   override / bonus ±1.0
"""

import logging
from typing import Optional


def _ema_trend(ema20, ema50, ema200, price) -> float:
    """Score EMA stack alignment: +1 full bull stack, -1 full bear stack."""
    score = 0.0
    if price and ema20 and ema50:
        if price > ema20 > ema50:
            score += 0.5
        elif price < ema20 < ema50:
            score -= 0.5
    if ema20 and ema50 and ema200:
        if ema20 > ema50 > ema200:
            score += 0.5
        elif ema20 < ema50 < ema200:
            score -= 0.5
    return score  # range -1 to +1


def _rsi_bias(rsi: float) -> float:
    """
    RSI bias scoring.
    50-70 = mild bullish, >70 = overbought warning (negative for new longs).
    30-50 = mild bearish, <30 = oversold warning (negative for new shorts).
    """
    if rsi is None:
        return 0.0
    if 50 < rsi <= 65:
        return 0.5
    elif rsi > 65:
        return -0.25  # overbought — mean reversion risk
    elif 35 <= rsi < 50:
        return -0.5
    elif rsi < 35:
        return 0.25   # oversold — mean reversion risk (flip)
    return 0.0


def _macd_direction(macd: float, signal: float) -> float:
    if macd is None or signal is None:
        return 0.0
    return 0.5 if macd > signal else -0.5


def _zscore_mean_reversion(z: float) -> float:
    """
    Z-score extremes ARE signals in themselves (mean reversion).
    z < -1.5 → bullish mean-reversion signal
    z > +1.5 → bearish mean-reversion signal
    """
    if z is None:
        return 0.0
    if z < -1.5:
        return min(abs(z) * 0.4, 1.0)
    elif z > 1.5:
        return -min(abs(z) * 0.4, 1.0)
    return 0.0


def _adx_modifier(adx: float) -> float:
    """Returns a strength multiplier for trend signals. Not a direction signal."""
    if adx is None:
        return 1.0
    if adx > 30:
        return 1.3   # strong trend — amplify directional signals
    elif adx < 20:
        return 0.7   # weak trend — reduce directional confidence
    return 1.0


def compute_confluence_score(
    latest_1h: dict,
    latest_4h: Optional[dict],
    latest_1d: Optional[dict],
    regime: Optional[dict] = None,
) -> dict:
    """
    Compute a multi-timeframe confluence score.

    Parameters
    ----------
    latest_1h, latest_4h, latest_1d : indicator dicts (row.to_dict())
    regime : output from regime_detector.detect_regime()

    Returns
    -------
    dict with keys: score, direction, strength, breakdown, recommendation
    """
    score = 0.0
    breakdown = []

    # --- 1H signals ---
    price_1h = latest_1h.get("close", 0)
    ema1h = _ema_trend(
        latest_1h.get("ema_20"), latest_1h.get("ema_50"),
        latest_1h.get("ema_200"), price_1h
    )
    weighted_1h = ema1h * 1.5
    score += weighted_1h
    breakdown.append(f"1H EMA stack: {'+' if weighted_1h > 0 else ''}{weighted_1h:.2f}")

    rsi1h = _rsi_bias(latest_1h.get("rsi"))
    score += rsi1h * 1.0
    breakdown.append(f"1H RSI({latest_1h.get('rsi', 0):.1f}) bias: {'+' if rsi1h > 0 else ''}{rsi1h:.2f}")

    macd1h = _macd_direction(latest_1h.get("macd"), latest_1h.get("macd_signal"))
    score += macd1h * 1.0
    breakdown.append(f"1H MACD direction: {'+' if macd1h > 0 else ''}{macd1h:.2f}")

    z = latest_1h.get("z_score", 0)
    zs = _zscore_mean_reversion(z)
    score += zs * 1.0
    breakdown.append(f"1H Z-Score({z:.2f}) mean-rev: {'+' if zs > 0 else ''}{zs:.2f}")

    adx_mult = _adx_modifier(latest_1h.get("adx"))

    # --- 4H signals (higher weight) ---
    if latest_4h:
        ema4h = _ema_trend(
            latest_4h.get("ema_20"), latest_4h.get("ema_50"),
            latest_4h.get("ema_200"), latest_4h.get("close")
        )
        weighted_4h = ema4h * 2.0
        score += weighted_4h
        breakdown.append(f"4H EMA stack: {'+' if weighted_4h > 0 else ''}{weighted_4h:.2f}")

        rsi4h = _rsi_bias(latest_4h.get("rsi"))
        score += rsi4h * 1.5
        breakdown.append(f"4H RSI({latest_4h.get('rsi', 0):.1f}) bias: {'+' if rsi4h > 0 else ''}{rsi4h * 1.5:.2f}")

    # --- 1D signals (highest weight) ---
    if latest_1d:
        ema1d = _ema_trend(
            latest_1d.get("ema_20"), latest_1d.get("ema_50"),
            latest_1d.get("ema_200"), latest_1d.get("close")
        )
        weighted_1d = ema1d * 2.0
        score += weighted_1d
        breakdown.append(f"1D EMA stack: {'+' if weighted_1d > 0 else ''}{weighted_1d:.2f}")

    # --- Apply ADX multiplier to the whole score ---
    score *= adx_mult
    breakdown.append(f"ADX({latest_1h.get('adx', 0):.1f}) strength multiplier: x{adx_mult:.2f}")

    # --- Regime gate ---
    if regime:
        regime_bias = regime.get("bias", "NEUTRAL")
        if regime_bias == "LONG" and score > 0:
            score += 1.0
            breakdown.append(f"Regime ({regime['label']}) confirms BULLISH: +1.00")
        elif regime_bias == "SHORT" and score < 0:
            score -= 1.0
            breakdown.append(f"Regime ({regime['label']}) confirms BEARISH: -1.00")
        elif regime_bias != "NEUTRAL" and (
            (regime_bias == "LONG" and score < 0) or
            (regime_bias == "SHORT" and score > 0)
        ):
            score *= 0.5  # Regime contradicts signal — halve confidence
            breakdown.append(f"Regime ({regime['label']}) CONTRADICTS signal — confidence halved")

    # --- Clamp to -10 / +10 ---
    score = max(-10.0, min(10.0, score))
    score = round(score, 2)

    # --- Interpret ---
    if score >= 4:
        direction = "BULLISH"
        strength = "HIGH"
        recommendation = "LONG entries valid — high MTF confluence"
    elif score >= 2:
        direction = "BULLISH"
        strength = "MODERATE"
        recommendation = "LONG bias — wait for 1H confirmation candle"
    elif score <= -4:
        direction = "BEARISH"
        strength = "HIGH"
        recommendation = "SHORT entries valid — high MTF confluence"
    elif score <= -2:
        direction = "BEARISH"
        strength = "MODERATE"
        recommendation = "SHORT bias — wait for 1H breakdown confirmation"
    else:
        direction = "NEUTRAL"
        strength = "LOW"
        recommendation = "No trade recommended — low MTF confluence"

    return {
        "score": score,
        "direction": direction,
        "strength": strength,
        "recommendation": recommendation,
        "breakdown": breakdown,
    }


def format_confluence_for_prompt(confluence: dict) -> str:
    """Format confluence dict into a readable string for AI injection."""
    lines = [
        f"MULTI-TIMEFRAME CONFLUENCE SCORE: {confluence['score']:+.2f} / 10.0",
        f"Direction: {confluence['direction']} | Strength: {confluence['strength']}",
        f"Recommendation: {confluence['recommendation']}",
        "Signal Breakdown:",
    ]
    for item in confluence["breakdown"]:
        lines.append(f"  • {item}")
    lines.append(
        "\nINSTRUCTION: If confluence strength is LOW (|score| < 2), you MUST output NEUTRAL "
        "regardless of individual indicator signals. Only enter trades when MTF confluence confirms direction."
    )
    return "\n".join(lines)

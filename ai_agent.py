"""
ai_agent.py  —  GoldAIAgent  (V4: Regime-Gated Ensemble Intelligence)

New in V4
---------
1. RegimeDetector integration
     - Market regime (TRENDING_BULL/BEAR, RANGING, VOLATILE) is computed
       before calling Gemini and injected into the prompt as hard context.
     - The LLM is instructed to align with the regime bias.

2. Confluence Score
     - build_confluence_score() aggregates RL signal + regime alignment +
       multi-timeframe RSI/MACD + Z-Score into a single 0-100 score.
     - Score and grade are injected into the prompt AND returned in the
       analysis response so paper_trader can gate position sizing.

3. Ensemble Voting Logic
     - RL and regime are now treated as INDEPENDENT votes.
     - If they disagree, the confluence score drops and the LLM is told
       to either stay neutral or reduce size.

4. Improved RL observation
     - Now uses the 15-feature observation from rl_env V4 to match the
       retrained model's expected input shape.
"""

import os
import json
import logging
import asyncio
import platform
import numpy as np
from stable_baselines3 import PPO

from regime_detector import get_current_regime, build_confluence_score

SYSTEM_PROMPT = """You are an expert financial market analyst and adaptive quantitative trading algorithm.
Your deep knowledge includes:
- Technical analysis (RSI, MACD, Fibonacci, Bollinger Bands, Elliott Wave)
- Institutional Quantitative Metrics:
  * VWAP: Price relative to VWAP indicates intraday institutional trend bias.
  * Z-Score: Measures statistical deviation from the mean (50 EMA). Z-score > 1.5 is heavily overbought, < -1.5 is heavily oversold. You MUST factor this into mean-reversion probabilities.
  * Volume Imbalance: Taker Buy/Sell delta EMA. Positive indicates aggressive market buying dominance; negative indicates aggressive selling.
  * Volatility Regime: ATR as a percentage of price to gauge market speed.
- Macroeconomic factors (real interest rates, inflation, USD dynamics, Fed policy)
- Geopolitical events and safe-haven demand dynamics
- Market sentiment, positioning data, and institutional flows

CRITICAL LEARNING INSTRUCTION:
You are provided with a "RECENT TRADE FEEDBACK" section containing your past trades.
You MUST analyze your past mistakes and successes.
- If a recent trade hit a STOP_LOSS, analyze your previous reasoning and identify what assumption was wrong. Adapt your current strategy to avoid repeating the mistake.
- If a recent trade hit TAKE_PROFIT, reinforce what indicators correctly predicted the move.
- In your "reasoning" output, you must explicitly state what you learned from the recent trades and how it influences your current prediction.
- DO NOT recommend entering trades that conflict with severe Z-Score extremes.

REGIME-GATED DECISION MAKING:
You will receive a MARKET REGIME section. This is mathematically computed from ADX and volatility.
- In TRENDING regimes: momentum signals have edge. Prioritize EMA alignment and MACD direction.
- In RANGING regime: mean-reversion signals have edge. Prioritize Z-Score and RSI extremes.
- In VOLATILE regime: REDUCE POSITION SIZE by 50%. Widen stops. Prefer NEUTRAL unless conviction is very high.
- NEVER fight the regime. A counter-regime trade requires explicit justification.

ENSEMBLE CONVICTION:
You will receive a CONFLUENCE SCORE (0-100) combining the RL signal, regime alignment, and multi-timeframe indicators.
- Score >= 75 (HIGH_CONVICTION): Full position size allowed.
- Score 55-74 (MODERATE): Reduce position to 0.7x.
- Score 35-54 (WEAK): Skip or use 0.3x size.
- Score < 35 (NO_EDGE): STAY NEUTRAL. The mathematical signals conflict.

Your analysis must always:
1. Consider multiple timeframes and quantitative extremes.
2. Weigh bullish and bearish factors objectively.
3. Factor in the lessons learned from recent trades.
4. Provide quantified confidence levels.
5. Explicitly respect the current market regime and confluence score.

NEVER provide guarantees. Always disclose limitations.
"""

TECHNICAL_ANALYSIS_PROMPT_TEMPLATE = """Analyze {self.symbol.upper()} using the following comprehensive data:

MACRO & CORRELATED ASSETS:
- DXY (US Dollar Index): {dxy}
- US10Y (Treasury Yield): {us10y}

MARKET REGIME (MATHEMATICALLY COMPUTED):
- Current Regime:       {regime}
- ADX Strength:         {regime_adx}
- Volatility Percentile:{regime_vol_pct}%
- Regime Confidence:    {regime_confidence}%
- Signal Bias:          {signal_bias}
- Strategy Hint:        {preferred_strategy}

MARKET DATA (MULTI-TIMEFRAME):
[1H TIMEFRAME]
- Current Price: {price_1h}
- RSI(14): {rsi_1h}
- MACD: {macd_1h} , Signal: {macd_sig_1h}
- EMA 20: {ema20_1h} , EMA 50: {ema50_1h} , EMA 200: {ema200_1h}
- ADX(14): {adx_1h}
- ATR(14): {atr_1h}

[4H TIMEFRAME]
- RSI(14): {rsi_4h}
- MACD: {macd_4h}
- EMA 20: {ema20_4h} , EMA 200: {ema200_4h}

[1D TIMEFRAME]
- RSI(14): {rsi_1d}
- EMA 50: {ema50_1d} , EMA 200: {ema200_1d}

ADVANCED INDICATORS (1H):
- Fibonacci Retracements (50-period): 0.382 at {fib382} OR 0.5 at {fib500} OR 0.618 at {fib618}
- Bollinger Bands: Upper {bb_upper} OR Middle {bb_middle} OR Lower {bb_lower}
- BB Width (squeeze indicator): {bb_width}%

INSTITUTIONAL QUANT METRICS (1H):
- VWAP (24-period): {vwap}
- VWAP Deviation: {vwap_dev_pct}%
- Z-Score (Price vs 50 EMA): {z_score}
- Volume Imbalance (Taker Buy/Sell EMA): {vol_imbalance}
- Volatility Regime: {vol_regime}%
- Realised Volatility (20-bar): {realized_vol}%
- Session Score (0=low liquidity, 1=London/NY overlap): {session_score}
- Real-Time Level 2 Order Book Imbalance: {ob_imbalance}

ENSEMBLE SIGNAL:
- RL Sniper Action:     {rl_action}
- Confluence Score:     {confluence_score}/100
- Conviction Grade:     {confluence_grade}
- Size Recommendation:  {confluence_rec}
- RL Aligned w/ Regime: {rl_aligned}
- Score Breakdown:      RL×Regime={score_rl} , MultiTF={score_tf} , RegimeConviction={score_reg} , ZScore={score_z}

RECENT TRADE FEEDBACK (LEARNING DATA):
{trade_history}

Provide analysis in JSON format exactly like the following schema.
CRITICAL: "target" and "stop_loss" MUST be a single float number. Do NOT provide ranges.
{{
  "trend": "uptrend OR downtrend OR sideways",
  "trend_strength": "1-10",
  "signals": {{"bullish": [], "bearish": []}},
  "regime_assessment": "Brief statement on why the current regime supports or contradicts the trade",
  "confluence_grade": "{confluence_grade}",
  "prediction": {{
    "4h":  {{"direction": "...", "target": "...", "stop_loss": "..."}},
    "24h": {{"direction": "...", "target": "...", "stop_loss": "..."}},
    "7d":  {{"direction": "...", "target": "..."}}
  }},
  "confidence": "0-100",
  "risk_factors": [],
  "reasoning": "Explain your technical analysis, how the regime gates your decision, how the ensemble score influenced sizing, and what you learned from recent trade feedback..."
}}
"""


class TradingAIAgent:
    def __init__(self, symbol='xauusdt'):
        logging.info("Initialized Gold AI Agent V6.0 (Regime-Gated Ensemble).")
        self.rl_model = None
        self.symbol = symbol
        model_path = f"models/ppo_{self.symbol}_final.zip"
        if os.path.exists(model_path):
            try:
                self.rl_model = PPO.load(model_path)
                logging.info(f"Loaded RL Sniper model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load RL model: {e}")

    # ── RL Inference ──────────────────────────────────────────────────────────

    def _get_rl_signal(self, latest_1h: dict) -> str:
        """Run RL model with V4 15-feature observation. Returns LONG/SHORT/NEUTRAL."""
        if not self.rl_model:
            return "NEUTRAL"
        try:
            from rl_env import AssetTradingEnv
            features = AssetTradingEnv.FEATURE_COLUMNS
            obs_vals = [float(latest_1h.get(col, 0.0)) for col in features]
            obs_vals.extend([0.0, 0.0, 0.0])  # position=0, current_risk_fraction=0, unrealised_pnl=0
            obs = np.nan_to_num(np.array(obs_vals, dtype=np.float32), nan=0.0)

            action, _ = self.rl_model.predict(obs, deterministic=True)
            
            raw_conviction = float(np.clip(action[0], -1.0, 1.0))
            conviction = round(raw_conviction * 10) / 10.0
            if abs(conviction) < 0.1:
                return "NEUTRAL"
            return "LONG" if conviction > 0 else "SHORT"
        except Exception as e:
            logging.error(f"RL inference error: {e}")
            return "NEUTRAL"

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def format_prompt(self, latest_1h, latest_4h, latest_1d,
                      recent_trades, macro_data,
                      regime_data: dict, confluence: dict,
                      rl_action: str) -> str:

        trade_history_str = "No recent trade history available."
        if recent_trades:
            lines = []
            for i, t in enumerate(recent_trades):
                outcome = "PROFIT" if t.get('pnl', 0) > 0 else "LOSS"
                lines.append(
                    f"Trade {i+1} [{outcome}]: Type={t.get('type')}, "
                    f"Entry={t.get('entry_price')}, Close={t.get('close_price')}, "
                    f"Reason={t.get('reason')}.\n"
                    f"Original Reasoning: {t.get('ai_reasoning')}"
                )
            trade_history_str = "\n\n".join(lines)

        comp = confluence.get('components', {})

        return TECHNICAL_ANALYSIS_PROMPT_TEMPLATE.format(
            dxy=macro_data.get('dxy', 'Unknown'),
            us10y=macro_data.get('us10y', 'Unknown'),

            # Regime
            regime=regime_data.get('regime', 'UNKNOWN'),
            regime_adx=regime_data.get('adx', 0),
            regime_vol_pct=regime_data.get('volatility_percentile', 50),
            regime_confidence=regime_data.get('confidence', 50),
            signal_bias=regime_data.get('signal_bias', 'MEAN_REVERSION'),
            preferred_strategy=regime_data.get('preferred_strategy', ''),

            # 1H
            price_1h=round(latest_1h.get('close', 0), 2),
            rsi_1h=round(latest_1h.get('rsi', 0), 2),
            macd_1h=round(latest_1h.get('macd', 0), 4),
            macd_sig_1h=round(latest_1h.get('macd_signal', 0), 4),
            ema20_1h=round(latest_1h.get('ema_20', 0), 2),
            ema50_1h=round(latest_1h.get('ema_50', 0), 2),
            ema200_1h=round(latest_1h.get('ema_200', 0), 2),
            adx_1h=round(latest_1h.get('adx', 0), 2),
            atr_1h=round(latest_1h.get('atr', 0), 2),

            # 4H
            rsi_4h=round(latest_4h.get('rsi', 0), 2) if latest_4h is not None else 'N/A',
            macd_4h=round(latest_4h.get('macd', 0), 4) if latest_4h is not None else 'N/A',
            ema20_4h=round(latest_4h.get('ema_20', 0), 2) if latest_4h is not None else 'N/A',
            ema200_4h=round(latest_4h.get('ema_200', 0), 2) if latest_4h is not None else 'N/A',

            # 1D
            rsi_1d=round(latest_1d.get('rsi', 0), 2) if latest_1d is not None else 'N/A',
            ema50_1d=round(latest_1d.get('ema_50', 0), 2) if latest_1d is not None else 'N/A',
            ema200_1d=round(latest_1d.get('ema_200', 0), 2) if latest_1d is not None else 'N/A',

            # Fibonacci
            fib382=round(latest_1h.get('fib_0.382', 0), 2),
            fib500=round(latest_1h.get('fib_0.500', 0), 2),
            fib618=round(latest_1h.get('fib_0.618', 0), 2),

            # Bollinger
            bb_upper=round(latest_1h.get('bb_upper', 0), 2),
            bb_middle=round(latest_1h.get('bb_middle', 0), 2),
            bb_lower=round(latest_1h.get('bb_lower', 0), 2),
            bb_width=round(latest_1h.get('bb_width', 0), 3),

            # Quant metrics
            vwap=round(latest_1h.get('vwap', 0), 2),
            vwap_dev_pct=round(latest_1h.get('vwap_dev_pct', 0), 3),
            z_score=round(latest_1h.get('z_score', 0), 2),
            vol_imbalance=round(latest_1h.get('vol_imbalance_ema', 0), 4),
            vol_regime=round(latest_1h.get('volatility_regime', 0), 4),
            realized_vol=round(latest_1h.get('realized_vol_20', 0), 2),
            session_score=round(latest_1h.get('session_score', 0.5), 2),
            ob_imbalance=round(latest_1h.get('order_book_imbalance', 0), 4),

            # Ensemble
            rl_action=rl_action,
            confluence_score=confluence.get('total_score', 0),
            confluence_grade=confluence.get('grade', 'UNKNOWN'),
            confluence_rec=confluence.get('recommendation', ''),
            rl_aligned=confluence.get('rl_aligned_with_regime', False),
            score_rl=comp.get('rl_regime_alignment', 0),
            score_tf=comp.get('multi_timeframe', 0),
            score_reg=comp.get('regime_conviction', 0),
            score_z=comp.get('z_score_confirmation', 0),

            trade_history=trade_history_str,
        )

    # ── Main Analysis Entry Point ─────────────────────────────────────────────

    async def analyze(self, latest_1h, latest_4h=None, latest_1d=None,
                      recent_trades=None, sentiment_data=None, macro_data=None,
                      hist_df=None):

        if recent_trades is None: recent_trades = []
        if macro_data    is None: macro_data    = {}

        # Step 1: Regime detection
        try:
            regime_data = get_current_regime(latest_1h, hist_df=hist_df)
        except Exception as e:
            logging.error(f"Regime detection failed: {e}")
            regime_data = {"regime": "RANGING", "adx": 0, "volatility_percentile": 50,
                           "confidence": 50, "signal_bias": "MEAN_REVERSION",
                           "preferred_strategy": "MEAN_REVERSION"}

        # Step 2: RL signal
        rl_action = self._get_rl_signal(latest_1h)

        # Step 3: Confluence score (ensemble voting)
        try:
            confluence = build_confluence_score(
                rl_action=rl_action,
                regime_data=regime_data,
                latest_1h=latest_1h,
                latest_4h=latest_4h,
                latest_1d=latest_1d,
            )
        except Exception as e:
            logging.error(f"Confluence scoring failed: {e}")
            confluence = {"total_score": 50, "grade": "MODERATE",
                          "recommendation": "EXECUTE — reduced size (0.7x)",
                          "rl_aligned_with_regime": False, "regime": "UNKNOWN",
                          "components": {}}

        logging.info(
            f"[V4 Ensemble] Regime={regime_data['regime']} | "
            f"RL={rl_action} | Confluence={confluence['total_score']} "
            f"({confluence['grade']})"
        )

        # Step 4: Load optimal params
        optimal_params_prompt = ""
        try:
            if os.path.exists("optimal_params.json"):
                with open("optimal_params.json") as f:
                    opt = json.load(f)
                optimal_params_prompt = (
                    f"\n\nBACKTEST OPTIMIZED GUIDELINES:\n"
                    f"Mathematically optimal entry parameters:\n"
                    f"- Z-Score Threshold: +/- {opt.get('z_score_threshold', 1.5)}\n"
                    f"- RSI Overbought: > {opt.get('rsi_overbought', 70)}\n"
                    f"- RSI Oversold:   < {opt.get('rsi_oversold', 30)}\n"
                    f"Heavily weigh these thresholds. If Z-Score and RSI are nowhere near "
                    f"these optimized levels, strongly consider staying neutral."
                )
        except Exception as e:
            logging.error(f"Could not load optimal_params: {e}")

        # Step 5: Sentiment
        sentiment_prompt = ""
        if sentiment_data:
            sentiment_prompt = (
                f" Market Sentiment: Fear & Greed = {sentiment_data.get('value')} "
                f"({sentiment_data.get('classification')}). "
                f"Factor this psychological context into your decision."
            )

        # Step 6: Build full prompt
        data_prompt = self.format_prompt(
            latest_1h, latest_4h, latest_1d,
            recent_trades, macro_data,
            regime_data, confluence, rl_action
        )

        full_prompt = (
            f"{SYSTEM_PROMPT}\n{data_prompt}"
            f"{sentiment_prompt}{optimal_params_prompt}"
        )
        full_prompt = full_prompt.replace("\r", "").replace("|", " OR ")

        # Step 7: Call Gemini
        cmd = "gemini.cmd" if platform.system() == "Windows" else "gemini"
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "ask", "--output-format", "json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=full_prompt.encode('utf-8'))

            if proc.returncode == 0:
                try:
                    cli_out  = json.loads(stdout.decode('utf-8'))
                except json.JSONDecodeError:
                    return json.dumps({"error": "CLI did not return JSON",
                                       "ensemble": confluence,
                                       "regime": regime_data})

                text = cli_out.get("response", "")
                s, e = text.find('{'), text.rfind('}')
                if s != -1 and e > s:
                    text = text[s:e+1]
                try:
                    parsed = json.loads(text)
                    # Attach ensemble metadata for downstream use
                    parsed['_ensemble'] = {
                        "regime":           regime_data['regime'],
                        "rl_action":        rl_action,
                        "confluence_score": confluence['total_score'],
                        "confluence_grade": confluence['grade'],
                        "size_multiplier":  self._size_multiplier(confluence['total_score'],
                                                                   regime_data['regime']),
                    }
                    return json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    return json.dumps({"error": "Invalid JSON from AI",
                                       "raw": text,
                                       "_ensemble": confluence})
            else:
                err = stderr.decode('utf-8')
                logging.error(f"Gemini CLI error (rc={proc.returncode}): {err}")
                return json.dumps({"error": "CLI failed", "stderr": err,
                                   "_ensemble": confluence})

        except FileNotFoundError:
            logging.error(f"Gemini CLI '{cmd}' not found.")
            return json.dumps({"error": "Gemini CLI not found", "_ensemble": confluence})
        except Exception as e:
            logging.error(f"Unexpected error calling Gemini: {e}")
            return json.dumps({"error": str(e), "_ensemble": confluence})

    @staticmethod
    def _size_multiplier(score: int, regime: str) -> float:
        """Convert confluence score + regime into a position size multiplier."""
        if regime == "VOLATILE":
            return 0.5
        if score >= 75: return 1.0
        if score >= 55: return 0.7
        if score >= 35: return 0.3
        return 0.0

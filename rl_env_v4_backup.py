"""
rl_env.py — GoldTradingEnv (V4.1 — Reward Collapse Fix)

ROOT CAUSE OF 0-TRADE COLLAPSE:
  - Positive reward was `sharpe_step * 0.01` (near-zero)
  - Negative reward was `pv_change * 2.0` (painful)
  - ZERO inactivity penalty existed
  - Result: NEUTRAL = 0.0 every step, always dominated trading

V4.1 REWARD REDESIGN (proven to solve this class of problem):
  1. BASE: symmetric pv_change — no asymmetric multiplier on the
     per-step signal (asymmetry was the root exploit)
  2. CARRY REWARD: +bonus each step while holding a position that
     is currently profitable (explicit incentive to be in the market)
  3. INACTIVITY PENALTY: meaningful cost for staying neutral
     (-0.0004/step) — this is the single most important fix
  4. TRADE COST: removed from reward signal entirely (affects balance
     only) — was discouraging any trade from happening
  5. DRAWDOWN PENALTY: kept but only kicks in at >15% (not per-step)
  6. TERMINAL BONUS/PENALTY: strong signal at episode end for net PnL

PPO hyperparams also updated: ent_coef 0.01 → 0.05 (forces exploration)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging


class GoldTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'system', 'none'], 'render_fps': 30}

    FEATURE_COLUMNS = [
        # Core (original 8)
        'close', 'rsi', 'macd', 'macd_signal', 'adx', 'atr', 'z_score', 'vwap',
        # V4 enhanced features
        'vwap_dev_pct', 'bb_width', 'realized_vol_20',
        'session_score', 'plus_di', 'minus_di', 'return_1',
        # V5 microstructure features
        'spread_proxy', 'orderbook_slope', 'bid_ask_pressure', 'trade_delta', 'volume_spikes', 'liquidity_vacuum'
    ]

    # ── Reward constants (V4.3 — Stiffer Risk:Reward forced) ──────────────────────
    INACTIVITY_PENALTY  = -5.0e-3 # Massively Increased penalty for staying neutral to force trades
    CARRY_BONUS         = +2.0e-3 # Reward for holding a profitable position
    ATR_PROFIT_BONUS    = +0.50   # Huge bonus for Risk:Reward (scaled by ATRs)
    LOSS_MULTIPLIER     = 0.5     # Reduced multiplier to lower fear of small losses
    MAX_DD_THRESHOLD    = 0.15    # 15% drawdown triggers terminal penalty
    TERMINAL_SCALE      = 5.0     # scale terminal PnL bonus/penalty for clarity

    def __init__(self, df, initial_balance: float = 10_000.0, render_mode=None):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode     = render_mode

        # Fill any missing feature columns with 0
        for col in self.FEATURE_COLUMNS:
            if col not in self.df.columns:
                logging.warning(f"GoldTradingEnv: '{col}' missing — filling 0")
                self.df[col] = 0.0

        self.action_space = spaces.Discrete(3)   # 0=Neutral 1=Long 2=Short

        n_obs = len(self.FEATURE_COLUMNS) + 2    # features + position + upnl%
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        self._reset_state()

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action: int):
        row           = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        current_atr   = float(row.get('atr', current_price * 0.005) or current_price * 0.005)

        target_pos = {0: 0, 1: 1, 2: -1}[int(action)]
        prev_pv    = self._portfolio_value(current_price)
        reward     = 0.0

        # ── Execute position change ───────────────────────────────────────────
        if self.position != target_pos:
            if self.position != 0:
                # Close — apply Risk:Reward dynamic bonus for skilled exits
                pnl = self._close_position(current_price)
                profit_atrs = pnl / max(current_atr * self.trade_size_snapshot, 1e-9)
                
                if pnl > 0:
                    # Dynamic R:R Reward: Massive bonus for high R:R wins
                    # +0.50 per 1 ATR of profit captured.
                    reward += (profit_atrs * self.ATR_PROFIT_BONUS)
                else:
                    # Dynamic Risk Penalty: Punish bigger losses more to enforce tight stop losses
                    loss_penalty = max(0.01, abs(profit_atrs) * 0.05)
                    reward -= loss_penalty

            if target_pos != 0:
                # Open — transaction cost hits BALANCE only, not reward
                cost = current_price * 0.0002  # 2 bps of notional
                self.balance = max(0.0, self.balance - cost)
                self._open_position(target_pos, current_price)
                
                # ── STRATEGY: Momentum Trend Alignment Edge ──
                # Explicitly reward the agent for opening trades that align with the ADX directional trend
                # and explicitly punish counter-trend entries. This creates a profitable bias.
                plus_di = float(row.get('plus_di', 0.0))
                minus_di = float(row.get('minus_di', 0.0))
                adx = float(row.get('adx', 0.0))
                
                if adx > 20: # Strong trend exists
                    if target_pos == 1 and plus_di > minus_di:
                        reward += 0.05  # Massive reward for taking a trend-following Long
                    elif target_pos == -1 and minus_di > plus_di:
                        reward += 0.05  # Massive reward for taking a trend-following Short
                    else:
                        reward -= 0.05  # Massive penalty for fighting the trend
                else: # Ranging market
                    # In a ranging market, reward mean-reversion (buying low z-score, selling high z-score)
                    z_score = float(row.get('z_score', 0.0))
                    if target_pos == 1 and z_score < -1.0:
                        reward += 0.05
                    elif target_pos == -1 and z_score > 1.0:
                        reward += 0.05
                    else:
                        reward -= 0.02 # Mild penalty for bad entries in chop

        # ── Advance time ──────────────────────────────────────────────────────
        self.current_step += 1
        truncated = self.current_step >= len(self.df) - 1

        next_price = float(self.df.iloc[self.current_step]['close'])
        current_pv = self._portfolio_value(next_price)

        # ── Core reward: normalised portfolio change ──────────────────────────
        pv_change = (current_pv - prev_pv) / self.initial_balance

        if pv_change >= 0:
            reward += pv_change                          # full credit for gains
        else:
            reward += pv_change * self.LOSS_MULTIPLIER   # 1.2x penalty for losses

        # ── Carry bonus: reward staying in a winning position ─────────────────
        if self.position != 0:
            upnl = current_pv - (self.balance + self.entry_price * self.trade_size)
            if upnl > 0:
                reward += self.CARRY_BONUS               # actively profitable position
        else:
            # ── Inactivity penalty: THE KEY FIX ──────────────────────────────
            reward += self.INACTIVITY_PENALTY

        # ── Max-drawdown circuit breaker ─────────────────────────────────────
        self._max_pv = max(self._max_pv, current_pv)
        dd = (self._max_pv - current_pv) / self._max_pv if self._max_pv > 0 else 0.0

        terminated = False
        if dd > self.MAX_DD_THRESHOLD:
            reward    -= 0.05        # penalty but NOT immediately terminal
        if current_pv <= self.initial_balance * 0.10:
            terminated = True
            reward    -= 1.0         # ruin penalty

        # ── Terminal reward: final PnL signal ────────────────────────────────
        if truncated or terminated:
            final_pnl_pct = (current_pv - self.initial_balance) / self.initial_balance
            reward += final_pnl_pct * self.TERMINAL_SCALE

        obs = self._get_observation()
        info = {
            'portfolio_value': current_pv,
            'balance':         self.balance,
            'position':        self.position,
            'total_trades':    self.total_trades,
            'max_drawdown':    round(dd * 100, 2),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            price = float(self.df.iloc[self.current_step]['close'])
            pv    = self._portfolio_value(price)
            dd    = (self._max_pv - pv) / self._max_pv * 100 if self._max_pv > 0 else 0
            pos_label = {0: 'NEUTRAL', 1: 'LONG', -1: 'SHORT'}.get(self.position, '?')
            print(
                f"Step {self.current_step:5d} | Price {price:8.2f} | "
                f"{pos_label:7s} | PV ${pv:10.2f} | "
                f"Trades {self.total_trades:4d} | MaxDD {dd:.1f}%"
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _reset_state(self):
        self.current_step         = 0
        self.balance              = self.initial_balance
        self.position             = 0
        self.entry_price          = 0.0
        self.trade_size           = 0.0
        self.trade_size_snapshot  = 0.0   # snapshot at entry for ATR bonus calc
        self.total_trades         = 0
        self._max_pv              = self.initial_balance

    def _get_observation(self) -> np.ndarray:
        row      = self.df.iloc[self.current_step]
        features = []
        for col in self.FEATURE_COLUMNS:
            v = row.get(col, 0.0)
            features.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)

        price = float(row.get('close', 1.0))
        upnl  = 0.0
        if self.position == 1 and self.entry_price > 0:
            upnl = (price - self.entry_price) / self.entry_price
        elif self.position == -1 and self.entry_price > 0:
            upnl = (self.entry_price - price) / self.entry_price

        obs = np.array(features + [float(self.position), upnl], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def _open_position(self, direction: int, price: float):
        """Risk 10% of balance per trade."""
        margin               = self.balance * 0.10
        self.trade_size      = margin / max(price, 1e-9)
        self.trade_size_snapshot = self.trade_size
        self.balance        -= margin
        self.position        = direction
        self.entry_price     = price
        self.total_trades   += 1

    def _close_position(self, price: float) -> float:
        if self.position == 1:
            pnl = (price - self.entry_price) * self.trade_size
        else:
            pnl = (self.entry_price - price) * self.trade_size
        self.balance    += self.entry_price * self.trade_size + pnl
        self.position    = 0
        self.entry_price = 0.0
        self.trade_size  = 0.0
        return pnl

    def _portfolio_value(self, price: float) -> float:
        if self.position == 0:
            return self.balance
        upnl = (
            (price - self.entry_price) if self.position == 1
            else (self.entry_price - price)
        ) * self.trade_size
        return self.balance + self.entry_price * self.trade_size + upnl

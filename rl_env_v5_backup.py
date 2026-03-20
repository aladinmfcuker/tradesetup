"""
rl_env.py — GoldTradingEnv (V5.0 — Pure ML: Sharpe + Profit Factor)

Upgrades:
- Removed hardcoded strategy edges (ADX, Z-Score).
- Introduces Rolling Sharpe and Profit Factor calculations.
- Combines dense PnL step rewards with a strong Terminal objective:
  `Reward = Base PnL + Terminal(Sharpe + Profit_Factor)`
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import math

class GoldTradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'system', 'none'], 'render_fps': 30}

    FEATURE_COLUMNS = [
        'close', 'rsi', 'macd', 'macd_signal', 'adx', 'atr', 'z_score', 'vwap',
        'vwap_dev_pct', 'bb_width', 'realized_vol_20',
        'session_score', 'plus_di', 'minus_di', 'return_1',
        'spread_proxy', 'orderbook_slope', 'bid_ask_pressure', 'trade_delta', 'volume_spikes', 'liquidity_vacuum'
    ]

    # ── Reward constants (V5.0) ───────────────────────────────────────────────
    INACTIVITY_PENALTY  = -1.0e-4 # Mild penalty to encourage finding trades
    CARRY_BONUS         = +1.0e-4 # Mild reward for riding winners
    MAX_DD_THRESHOLD    = 0.15    # 15% drawdown triggers terminal penalty
    TERMINAL_SCALE      = 2.0     # Scale for the Sharpe + PF bonus

    def __init__(self, df, initial_balance: float = 10_000.0, render_mode=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        for col in self.FEATURE_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = 0.0

        self.action_space = spaces.Discrete(3)
        n_obs = len(self.FEATURE_COLUMNS) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self._reset_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        
        target_pos = {0: 0, 1: 1, 2: -1}[int(action)]
        prev_pv = self._portfolio_value(current_price)
        reward = 0.0

        # Execute position change
        if self.position != target_pos:
            if self.position != 0:
                pnl = self._close_position(current_price)
                if pnl > 0:
                    self.gross_profit += pnl
                else:
                    self.gross_loss += abs(pnl)

            if target_pos != 0:
                cost = current_price * 0.0002
                self.balance = max(0.0, self.balance - cost)
                self._open_position(target_pos, current_price)

        self.current_step += 1
        truncated = self.current_step >= len(self.df) - 1
        next_price = float(self.df.iloc[self.current_step]['close'])
        current_pv = self._portfolio_value(next_price)

        # Step Returns for Sharpe
        step_ret = (current_pv - prev_pv) / max(prev_pv, 1e-9)
        self.returns_history.append(step_ret)

        # Core dense reward: Pv change
        reward += step_ret 

        # Carry / Inactivity
        if self.position != 0:
            upnl = current_pv - (self.balance + self.entry_price * self.trade_size)
            if upnl > 0:
                reward += self.CARRY_BONUS
        else:
            reward += self.INACTIVITY_PENALTY

        # Max Drawdown
        self._max_pv = max(self._max_pv, current_pv)
        dd = (self._max_pv - current_pv) / self._max_pv if self._max_pv > 0 else 0.0

        terminated = False
        if dd > self.MAX_DD_THRESHOLD:
            reward -= 0.1
        if current_pv <= self.initial_balance * 0.10:
            terminated = True
            reward -= 1.0

        # ── TERMINAL REWARD: Sharpe + Profit Factor ──
        if truncated or terminated:
            pf = self.gross_profit / max(self.gross_loss, 1e-9)
            
            returns_array = np.array(self.returns_history)
            std = np.std(returns_array)
            sharpe = 0.0
            if std > 0:
                sharpe = np.mean(returns_array) / std * math.sqrt(252 * 24)
            
            # Bound the metrics to prevent exploding gradients
            pf = min(pf, 5.0) 
            sharpe = min(max(sharpe, -2.0), 5.0)

            # Only reward terminal metrics if we actually took trades
            if self.total_trades > 0:
                terminal_bonus = (sharpe + pf) * self.TERMINAL_SCALE
                reward += terminal_bonus

        obs = self._get_observation()
        info = {
            'portfolio_value': current_pv,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'max_drawdown': round(dd * 100, 2),
            'sharpe': sharpe if (truncated or terminated) and self.total_trades > 0 else 0.0,
            'profit_factor': pf if (truncated or terminated) and self.total_trades > 0 else 0.0
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            price = float(self.df.iloc[self.current_step]['close'])
            pv = self._portfolio_value(price)
            dd = (self._max_pv - pv) / self._max_pv * 100 if self._max_pv > 0 else 0
            pos_label = {0: 'NEUTRAL', 1: 'LONG', -1: 'SHORT'}.get(self.position, '?')
            print(
                f"Step {self.current_step:5d} | Price {price:8.2f} | "
                f"{pos_label:7s} | PV ${pv:10.2f} | "
                f"Trades {self.total_trades:4d} | MaxDD {dd:.1f}%"
            )

    def _reset_state(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_size = 0.0
        self.total_trades = 0
        self._max_pv = self.initial_balance
        
        # New trackers for metrics
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.returns_history = []

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        features = []
        for col in self.FEATURE_COLUMNS:
            v = row.get(col, 0.0)
            features.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)

        price = float(row.get('close', 1.0))
        upnl = 0.0
        if self.position == 1 and self.entry_price > 0:
            upnl = (price - self.entry_price) / self.entry_price
        elif self.position == -1 and self.entry_price > 0:
            upnl = (self.entry_price - price) / self.entry_price

        obs = np.array(features + [float(self.position), upnl], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

    def _open_position(self, direction: int, price: float):
        margin = self.balance * 0.10
        self.trade_size = margin / max(price, 1e-9)
        self.balance -= margin
        self.position = direction
        self.entry_price = price
        self.total_trades += 1

    def _close_position(self, price: float) -> float:
        if self.position == 1:
            pnl = (price - self.entry_price) * self.trade_size
        else:
            pnl = (self.entry_price - price) * self.trade_size
        self.balance += self.entry_price * self.trade_size + pnl
        self.position = 0
        self.entry_price = 0.0
        self.trade_size = 0.0
        return pnl

    def _portfolio_value(self, price: float) -> float:
        if self.position == 0:
            return self.balance
        upnl = (
            (price - self.entry_price) if self.position == 1
            else (self.entry_price - price)
        ) * self.trade_size
        return self.balance + self.entry_price * self.trade_size + upnl

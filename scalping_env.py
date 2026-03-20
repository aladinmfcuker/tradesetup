import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class ScalpingTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for high-frequency scalping based on Level 2 Order Book Imbalance.
    """
    metadata = {'render_modes': ['human', 'system', 'none'], 'render_fps': 30}

    def __init__(self, df, initial_balance=10000.0, render_mode=None):
        super(ScalpingTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode
        
        # Features heavily focused on immediate liquidity
        # Need to ensure these are calculated in indicators.py
        self.feature_columns = ['close', 'rsi', 'macd', 'atr', 'vol_imbalance_ema', 'ob_imbalance']
        
        # Action Space: 0 = Neutral, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: Features + Position + Unrealized PnL %
        num_features = len(self.feature_columns) + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        # Internal state
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_size = 0.0
        self.total_trades = 0
        
        # Risk Management for Scalping (Asymmetric Reward Function)
        self.take_profit_pct = 0.002 # 0.2% TP
        self.stop_loss_pct = 0.001   # 0.1% SL (Risk 1 to make 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_size = 0.0
        self.total_trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        features = []
        for col in self.feature_columns:
            val = row.get(col, 0.0)
            if pd.isna(val): val = 0.0
            features.append(val)
            
        current_price = row['close']
        unrealized_pnl_pct = 0.0
        if self.position == 1:
            unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        obs = np.array(features + [self.position, unrealized_pnl_pct], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
            
        reward = 0.0
        step_penalty = -0.0001 # Small penalty for doing nothing to encourage finding trades
        
        # Check if current position hit SL or TP before processing new action
        if self.position != 0:
            unrealized_pnl_pct = 0
            if self.position == 1:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            elif self.position == -1:
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
                
            if unrealized_pnl_pct >= self.take_profit_pct or unrealized_pnl_pct <= -self.stop_loss_pct:
                # Force close position
                target_position = 0 
                # Massive reward for hitting TP, massive penalty for SL
                if unrealized_pnl_pct >= self.take_profit_pct:
                    reward += 1.0 # Big success
                else:
                    reward -= 1.0 # Big failure

        # Execute trades
        if self.position != target_position:
            # Close existing
            if self.position != 0:
                pnl = 0
                if self.position == 1:
                    pnl = (current_price - self.entry_price) * self.trade_size
                elif self.position == -1:
                    pnl = (self.entry_price - current_price) * self.trade_size
                
                margin_used = self.entry_price * self.trade_size
                self.balance += (margin_used + pnl)
                self.position = 0
                self.entry_price = 0.0
                self.trade_size = 0.0
                
            # Open new
            if target_position != 0:
                self.position = target_position
                self.entry_price = current_price
                margin_to_use = self.balance * 0.10 # Risk 10% of portfolio
                self.trade_size = margin_to_use / current_price
                self.balance -= margin_to_use
                self.total_trades += 1
                
        if self.position == 0:
            reward += step_penalty

        self.current_step += 1
        
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.df) - 1:
            truncated = True
            
        current_portfolio_value = self._get_portfolio_value(current_price)
        if current_portfolio_value <= self.initial_balance * 0.5: # 50% DD limit
            terminated = True
            reward -= 10.0 # Huge penalty for blowing up account
            
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _get_portfolio_value(self, current_price):
        if self.position == 0: return self.balance
        unrealized_pnl = 0
        if self.position == 1: unrealized_pnl = (current_price - self.entry_price) * self.trade_size
        elif self.position == -1: unrealized_pnl = (self.entry_price - current_price) * self.trade_size
        return self.balance + (self.entry_price * self.trade_size) + unrealized_pnl

    def render(self):
        pass

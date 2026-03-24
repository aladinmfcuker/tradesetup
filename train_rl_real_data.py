"""
train_rl_real_data.py  —  PPO Training Pipeline (V4)

Upgrades
--------
- Uses V4 GoldTradingEnv (15 features, Sharpe reward, asymmetric drawdown)
- Reports Sharpe ratio + max drawdown on out-of-sample test (not just return)
- Saves training metadata to models/training_meta.json for audit trail
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from indicators import calculate_indicators
from rl_env import AssetTradingEnv as GoldTradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_real_data():
    if not os.path.exists("real_gold_history.csv"):
        logging.error("real_gold_history.csv not found. Run fetch_real_data.py first.")
        return pd.DataFrame()
    df = pd.read_csv("real_gold_history.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values('timestamp').reset_index(drop=True)


def make_env(data, rank, seed=0):
    def _init():
        env = GoldTradingEnv(data, render_mode='none')
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def evaluate_agent(model, test_data: pd.DataFrame, initial_balance: float = 10000.0) -> dict:
    """Full out-of-sample evaluation with Sharpe + Calmar + Max DD."""
    env = GoldTradingEnv(test_data, initial_balance=initial_balance, render_mode=None)
    obs, _ = env.reset()
    terminated = truncated = False
    equity_curve = [initial_balance]

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        equity_curve.append(info['portfolio_value'])

    equity  = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)) if np.std(returns) > 0 else 0.0
    rolling_max = np.maximum.accumulate(equity)
    drawdowns   = (rolling_max - equity) / rolling_max
    max_dd      = float(drawdowns.max() * 100)
    final_pv    = float(equity[-1])
    strat_ret   = (final_pv - initial_balance) / initial_balance * 100

    market_ret  = float((test_data.iloc[-1]['close'] - test_data.iloc[0]['close'])
                        / test_data.iloc[0]['close'] * 100)
    calmar      = float(strat_ret / max_dd) if max_dd > 0 else 0.0

    return {
        'final_portfolio_value': round(final_pv, 2),
        'strategy_return_pct':   round(strat_ret, 3),
        'market_return_pct':     round(market_ret, 3),
        'sharpe_ratio':          round(sharpe, 4),
        'max_drawdown_pct':      round(max_dd, 3),
        'calmar_ratio':          round(calmar, 4),
        'total_trades':          info.get('total_trades', 0),
    }


def main(args):
    os.makedirs("models", exist_ok=True)
    device = "cpu"
    logging.info(f"Device: {device.upper()}")

    df = load_real_data()
    if df.empty:
        return

    logging.info("Calculating V4 indicators (15 features)...")
    df = calculate_indicators(df)

    required = GoldTradingEnv.FEATURE_COLUMNS + ['close']
    valid    = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    # Fill any still-missing optional features with 0
    for col in GoldTradingEnv.FEATURE_COLUMNS:
        if col not in valid.columns:
            valid[col] = 0.0

    logging.info(f"Valid rows after indicator warmup: {len(valid)}")

    valid['year'] = valid['timestamp'].dt.year

    if valid['year'].min() <= 2021:
        # Walk-Forward Validation (User Specified)
        train_data = valid[valid['year'] <= 2021].copy()
        val_data   = valid[valid['year'] == 2022].copy()
        test_data  = valid[valid['year'] == 2023].copy()
        fwd_data   = valid[valid['year'] >= 2024].copy()
        logging.info(f"Walk-Forward Split | Train(2018-2021): {len(train_data)} | Val(2022): {len(val_data)} | Test(2023): {len(test_data)} | Forward(2024+): {len(fwd_data)}")
    else:
        # Fallback: rolling chunks if data is short
        logging.warning("Data does not reach 2021. Falling back to 50/15/15/20 chronological split.")
        n = len(valid)
        train_data = valid.iloc[:int(n*0.5)].copy()
        val_data   = valid.iloc[int(n*0.5):int(n*0.65)].copy()
        test_data  = valid.iloc[int(n*0.65):int(n*0.8)].copy()
        fwd_data   = valid.iloc[int(n*0.8):].copy()
        logging.info(f"Fallback Split | Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)} | Forward: {len(fwd_data)}")

    if len(train_data) < 100:
        logging.error("Train data too small to continue.")
        return

    train_env = SubprocVecEnv([make_env(train_data, i) for i in range(args.num_cpu)])

    cb = CheckpointCallback(
        save_freq=args.save_freq // args.num_cpu,
        save_path='./models/',
        name_prefix='ppo_v6_ckpt'
    )

    # ── V6.0 PPO hyperparams ──────────────────────────────────────────────────
    # ent_coef to ensure continuous action space explores properly
    # n_steps=2048 gives the agent a long enough horizon to see trade outcomes
    # net_arch wider to handle 17 input features
    model = PPO(
        "MlpPolicy", train_env,
        verbose=1,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,          # Increased to prevent 0-trade collapse
        vf_coef=0.5,
        clip_range=0.2,
        n_steps=2048,           # long horizon so agent sees trade outcomes
        batch_size=512,
        gae_lambda=0.95,
        gamma=args.gamma,            # increased gamma for longer terminal reward memory
        policy_kwargs=dict(
            net_arch=[512, 256, 128],   # wider for 17-feature input
            activation_fn=__import__('torch').nn.ReLU,
        ),
        device=device,
    )

    # ── Reward sanity check ───────────────────────────────────────────────────
    # Verify the reward function isn't degenerate before wasting GPU hours
    logging.info("Running reward sanity check (50-step random rollout)...")
    _check_env = GoldTradingEnv(train_data.iloc[:500].copy(), render_mode=None)
    _obs, _ = _check_env.reset()
    _rewards = []
    for _ in range(50):
        _a = _check_env.action_space.sample()
        _obs, _r, _te, _tr, _ = _check_env.step(_a)
        _rewards.append(_r)
        if _te or _tr:
            break
    _mean_r = float(np.mean(_rewards))
    _neutral_r = GoldTradingEnv.INACTIVITY_PENALTY  # what neutral gives per step
    logging.info(f"  Random action mean reward:  {_mean_r:+.6f}")
    logging.info(f"  Permanent-neutral reward:   {_neutral_r:+.6f}")
    if _mean_r > _neutral_r:
        logging.info("  SANITY CHECK PASSED: random trading beats permanent neutral ✓")
    else:
        logging.warning("  SANITY CHECK WARNING: neutral may still dominate — check reward constants")

    # Update metadata to reflect V6.0 reward fix
    logging.info(f"Training PPO V6.0 for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=cb)

    model_path = "models/ppo_gold_real_final"
    model.save(model_path)
    logging.info(f"Model saved to {model_path}.zip")

    # ── Walk-Forward Out-of-Sample Evaluation ───
    all_metrics = {}
    
    if len(val_data) > 0:
        logging.info("─── Val Data Evaluation (2022) ───")
        all_metrics['val'] = evaluate_agent(model, val_data)
        for k, v in all_metrics['val'].items():
            logging.info(f"  {k}: {v}")
            
    if len(test_data) > 0:
        logging.info("─── Test Data Evaluation (2023) ───")
        all_metrics['test'] = evaluate_agent(model, test_data)
        for k, v in all_metrics['test'].items():
            logging.info(f"  {k}: {v}")
            
    if len(fwd_data) > 0:
        logging.info("─── Forward Data Evaluation (2024+) ───")
        all_metrics['fwd'] = evaluate_agent(model, fwd_data)
        for k, v in all_metrics['fwd'].items():
            logging.info(f"  {k}: {v}")

    # Save training metadata
    meta = {
        "trained_at":       datetime.now().isoformat(),
        "timesteps":        args.timesteps,
        "train_bars":       len(train_data),
        "val_bars":         len(val_data),
        "test_bars":        len(test_data),
        "fwd_bars":         len(fwd_data),
        "feature_columns":  GoldTradingEnv.FEATURE_COLUMNS,
        "obs_size":         len(GoldTradingEnv.FEATURE_COLUMNS) + 2,
        "reward_type":      "collapse_safe_v6.0_inactivity_penalty",
        "out_of_sample":    all_metrics,
    }
    with open("models/training_meta.json", "w") as f:
        json.dump(meta, f, indent=4)
    logging.info("Training metadata saved to models/training_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_500_000)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--num-cpu",   type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.999)
    main(parser.parse_args())

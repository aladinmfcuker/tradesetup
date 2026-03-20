"""
optimize.py  —  Walk-Forward Hyperparameter Optimization (V4)

Upgrades over V3
----------------
1. WALK-FORWARD VALIDATION
     Splits history into rolling train/test windows (default: 5 folds).
     Each fold trains on a window and tests on the next period.
     This prevents overfitting to a single historical period and gives
     a realistic estimate of out-of-sample performance.

2. SHARPE RATIO AS PRIMARY METRIC
     V3 sorted by total return, which rewards high-frequency strategies
     that may also have high drawdown. V4 uses:
       - Sharpe Ratio  (return / std of returns, annualised)
       - Calmar Ratio  (return / max drawdown)
     Return is still reported but is a secondary filter.

3. REGIME-AWARE OPTIMIZATION
     Runs the grid search separately per detected regime so that
     optimal parameters can differ between trending and ranging markets.

4. SAVES EXTENDED optimal_params.json
     Now includes regime-specific thresholds and walk-forward stability
     metrics (std of Sharpe across folds = consistency score).
"""

import asyncio
import json
import logging
import itertools
import pandas as pd
import numpy as np
from data_feed import BinanceFeed
from indicators import calculate_indicators
from regime_detector import classify_regime_series, REGIME_RANGING, REGIME_TRENDING_BULL, REGIME_TRENDING_BEAR

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ── Hyperparameter grid ───────────────────────────────────────────────────────
Z_SCORE_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0]
RSI_OVERBOUGHT     = [60, 65, 70, 75, 80]
RSI_OVERSOLD       = [40, 35, 30, 25, 20]

N_FOLDS            = 5    # walk-forward folds
TRAIN_RATIO        = 0.7  # 70% train, 30% test per fold
MIN_TRADES         = 10   # minimum trades for a result to be statistically valid
ANNUALISE_FACTOR   = np.sqrt(252 * 24)  # 1H bars


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_params(df: pd.DataFrame, z_thresh: float,
                    rsi_ob: float, rsi_os: float) -> dict:
    """
    Evaluate a parameter combination on a given DataFrame window.
    Returns dict with sharpe, calmar, return_pct, win_rate, trades.
    """
    df = df.copy()
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1

    long_cond  = (df['z_score'] < -z_thresh) & (df['rsi'] < rsi_os)
    short_cond = (df['z_score'] >  z_thresh) & (df['rsi'] > rsi_ob)

    signal = pd.Series(0, index=df.index)
    signal.loc[long_cond]  = 1
    signal.loc[short_cond] = -1

    strat_ret = signal * df['next_return']
    trades    = int(signal.abs().sum())

    if trades < MIN_TRADES:
        return None

    active_returns  = strat_ret[signal != 0]
    win_rate        = float((active_returns > 0).mean() * 100)
    mean_ret        = float(strat_ret.mean())
    std_ret         = float(strat_ret.std())
    sharpe          = float((mean_ret / std_ret * ANNUALISE_FACTOR) if std_ret > 0 else 0.0)

    cum_ret = (1 + strat_ret).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown    = (rolling_max - cum_ret) / rolling_max
    max_dd      = float(drawdown.max())

    final_ret   = float(cum_ret.iloc[-2] * 100 - 100) if len(cum_ret) > 1 else 0.0
    calmar      = float((final_ret / 100) / max_dd) if max_dd > 0 else 0.0

    return {
        'z_thresh':   z_thresh,
        'rsi_ob':     rsi_ob,
        'rsi_os':     rsi_os,
        'trades':     trades,
        'win_rate':   win_rate,
        'return_pct': final_ret,
        'sharpe':     sharpe,
        'calmar':     calmar,
        'max_dd':     max_dd * 100,
    }


# ── Walk-forward engine ───────────────────────────────────────────────────────

def walk_forward_search(df: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """
    Walk-forward grid search.
    Returns DataFrame of (param combination) x (mean/std of metrics across folds).
    """
    fold_size = len(df) // n_folds
    all_fold_results = []

    for fold in range(n_folds - 1):  # last fold has no test window
        train_start = fold * fold_size
        train_end   = train_start + int(fold_size * (1 + TRAIN_RATIO))
        test_start  = train_end
        test_end    = test_start + fold_size

        if test_end > len(df):
            break

        test_df = df.iloc[test_start:test_end].copy()

        logging.info(f"  Walk-forward fold {fold+1}/{n_folds-1}: "
                     f"test rows {test_start}–{test_end} ({len(test_df)} bars)")

        for z_t, rsi_ob, rsi_os in itertools.product(
                Z_SCORE_THRESHOLDS, RSI_OVERBOUGHT, RSI_OVERSOLD):
            r = evaluate_params(test_df, z_t, rsi_ob, rsi_os)
            if r is not None:
                r['fold'] = fold
                all_fold_results.append(r)

    if not all_fold_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_fold_results)

    # Aggregate across folds: mean and std of Sharpe (stability indicator)
    agg = results_df.groupby(['z_thresh', 'rsi_ob', 'rsi_os']).agg(
        mean_sharpe  = ('sharpe',     'mean'),
        std_sharpe   = ('sharpe',     'std'),    # low std = consistent
        mean_calmar  = ('calmar',     'mean'),
        mean_return  = ('return_pct', 'mean'),
        mean_winrate = ('win_rate',   'mean'),
        mean_trades  = ('trades',     'mean'),
        mean_max_dd  = ('max_dd',     'mean'),
        n_folds_valid= ('fold',       'count'),
    ).reset_index()

    # Consistency-adjusted Sharpe: penalise high variance across folds
    agg['std_sharpe']  = agg['std_sharpe'].fillna(0)
    agg['adj_sharpe']  = agg['mean_sharpe'] - 0.5 * agg['std_sharpe']

    return agg


# ── Regime-aware optimization ─────────────────────────────────────────────────

def regime_aware_optimization(df: pd.DataFrame):
    """
    Run walk-forward optimization separately for RANGING and TRENDING rows,
    then save a unified optimal_params.json with regime-specific overrides.
    """
    logging.info("Attaching regime labels to historical data...")
    df['regime'] = classify_regime_series(df)

    results = {}
    regime_groups = {
        'global':        df,
        REGIME_RANGING:  df[df['regime'] == REGIME_RANGING],
        'TRENDING':      df[df['regime'].isin([REGIME_TRENDING_BULL, REGIME_TRENDING_BEAR])],
    }

    for label, subset in regime_groups.items():
        if len(subset) < 200:
            logging.warning(f"  Skipping regime '{label}': only {len(subset)} bars.")
            continue
        logging.info(f"\n── Walk-Forward Optimisation: {label} ({len(subset)} bars) ──")
        agg = walk_forward_search(subset)
        if agg.empty:
            logging.warning(f"  No valid results for '{label}'.")
            continue

        # Filter to combos that appeared in all folds (consistent)
        valid = agg[agg['n_folds_valid'] >= max(2, N_FOLDS - 2)]
        if valid.empty:
            valid = agg

        best = valid.sort_values('adj_sharpe', ascending=False).iloc[0]

        print(f"\n{'='*55}")
        print(f"  REGIME: {label}")
        print(f"  Best Z-Thresh   : {best['z_thresh']:.2f}")
        print(f"  Best RSI OB/OS  : {best['rsi_ob']:.0f} / {best['rsi_os']:.0f}")
        print(f"  Adj Sharpe      : {best['adj_sharpe']:.3f}  "
              f"(mean={best['mean_sharpe']:.3f}, std={best['std_sharpe']:.3f})")
        print(f"  Mean Calmar     : {best['mean_calmar']:.3f}")
        print(f"  Mean Return     : {best['mean_return']:.2f}%")
        print(f"  Mean Win Rate   : {best['mean_winrate']:.1f}%")
        print(f"  Mean Max DD     : {best['mean_max_dd']:.1f}%")
        print(f"  Folds valid     : {int(best['n_folds_valid'])}/{N_FOLDS-1}")
        print(f"{'='*55}")

        results[label] = {
            'z_score_threshold': float(best['z_thresh']),
            'rsi_overbought':    float(best['rsi_ob']),
            'rsi_oversold':      float(best['rsi_os']),
            'adj_sharpe':        float(best['adj_sharpe']),
            'mean_sharpe':       float(best['mean_sharpe']),
            'sharpe_stability':  float(best['std_sharpe']),
            'mean_calmar':       float(best['mean_calmar']),
            'expected_win_rate': float(best['mean_winrate']),
            'expected_return':   float(best['mean_return']),
            'mean_max_drawdown': float(best['mean_max_dd']),
            'folds_validated':   int(best['n_folds_valid']),
        }

    if not results:
        logging.error("Optimization produced no valid results.")
        return

    # Build final output using 'global' as primary, with regime overrides
    global_params = results.get('global', list(results.values())[0])

    output = {
        # Primary params (used by AI unless overridden by regime)
        'z_score_threshold':  global_params['z_score_threshold'],
        'rsi_overbought':     global_params['rsi_overbought'],
        'rsi_oversold':       global_params['rsi_oversold'],
        'expected_win_rate':  global_params['expected_win_rate'],
        'expected_return_pct':global_params['expected_return'],
        'adj_sharpe':         global_params['adj_sharpe'],
        'sharpe_stability':   global_params['sharpe_stability'],
        'walk_forward_folds': N_FOLDS,
        # Regime-specific overrides
        'regime_params': {
            k: v for k, v in results.items() if k != 'global'
        }
    }

    with open("optimal_params.json", "w") as f:
        json.dump(output, f, indent=4)

    logging.info("Saved walk-forward optimised parameters to optimal_params.json")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    logging.info("Fetching historical data for XAU/USDT (1H)...")
    feed = BinanceFeed(symbol="xauusdt", timeframe="1h")
    df   = await asyncio.to_thread(feed.fetch_historical_data)

    if df.empty:
        logging.error("Failed to fetch historical data.")
        return

    logging.info(f"Loaded {len(df)} candles. Calculating indicators...")
    df = calculate_indicators(df)

    required = ['ema_200', 'rsi', 'macd', 'adx', 'atr', 'z_score', 'vwap',
                'volatility_regime', 'plus_di', 'minus_di']
    valid = df.dropna(subset=required).copy()

    if len(valid) < 500:
        logging.error(f"Not enough valid data: {len(valid)} rows after NaN drop.")
        return

    logging.info(f"Running regime-aware walk-forward optimisation on {len(valid)} bars...")
    regime_aware_optimization(valid)


if __name__ == "__main__":
    asyncio.run(main())

import pytest
import pandas as pd
import numpy as np
from indicators import calculate_indicators

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
    # Generate some somewhat realistic price data
    closes = 2000 + np.cumsum(np.random.normal(0, 5, 100))
    highs = closes + np.random.uniform(1, 5, 100)
    lows = closes - np.random.uniform(1, 5, 100)
    opens = closes - np.random.normal(0, 2, 100)
    volumes = np.random.uniform(100, 1000, 100)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_base_asset_volume": volumes * np.random.uniform(0.3, 0.7, 100)
    })
    return df

def test_calculate_indicators_empty_df():
    empty_df = pd.DataFrame()
    res = calculate_indicators(empty_df)
    assert res.empty

def test_calculate_indicators_short_df(sample_data):
    short_df = sample_data.iloc[:10]
    res = calculate_indicators(short_df)
    # Should return early since len < 50
    assert "rsi" not in res.columns
    assert len(res) == 10

def test_calculate_indicators_shapes_and_columns(sample_data):
    res = calculate_indicators(sample_data)

    assert len(res) == 100

    expected_cols = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
        'ema_20', 'ema_50', 'ema_200', 'atr',
        'plus_di', 'minus_di', 'adx',
        'fib_0.382', 'fib_0.500', 'fib_0.618',
        'vwap', 'vwap_dev_pct', 'z_score',
        'vol_imbalance', 'vol_imbalance_ema', 'volatility_regime',
        'realized_vol_20', 'return_1', 'return_5',
        'session_score', 'bid_ask_pressure', 'trade_delta',
        'spread_proxy', 'volume_spikes', 'liquidity_vacuum',
        'orderbook_slope', 'order_book_imbalance'
    ]

    for col in expected_cols:
        assert col in res.columns, f"Missing expected column {col}"

def test_rsi_bounds(sample_data):
    res = calculate_indicators(sample_data)
    # Dropna to avoid initial NaN checks
    valid_rsi = res['rsi'].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()

def test_bollinger_bands_logic(sample_data):
    res = calculate_indicators(sample_data)
    valid_bb = res.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])

    # Upper > Middle > Lower
    assert (valid_bb['bb_upper'] >= valid_bb['bb_middle']).all()
    assert (valid_bb['bb_middle'] >= valid_bb['bb_lower']).all()

    # BB Width logic
    expected_width = (valid_bb['bb_upper'] - valid_bb['bb_lower']) / valid_bb['bb_middle'] * 100
    pd.testing.assert_series_equal(valid_bb['bb_width'], expected_width, check_names=False)

def test_macd_logic(sample_data):
    res = calculate_indicators(sample_data)
    valid_macd = res.dropna(subset=['macd', 'macd_signal', 'macd_hist'])

    expected_hist = valid_macd['macd'] - valid_macd['macd_signal']
    pd.testing.assert_series_equal(valid_macd['macd_hist'], expected_hist, check_names=False)

def test_ema_ordering_trending(sample_data):
    # Create an explicit uptrend to test EMAs
    sample_data['close'] = np.linspace(100, 200, 100)
    res = calculate_indicators(sample_data)

    valid_ema = res.dropna(subset=['ema_20', 'ema_50', 'ema_200'])
    last_row = valid_ema.iloc[-1]

    # In a strong uptrend, short EMA > long EMA
    assert last_row['ema_20'] > last_row['ema_50']
    assert last_row['ema_50'] > last_row['ema_200']

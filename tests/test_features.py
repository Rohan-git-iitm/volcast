"""Check that features use only the past and the target uses only the future."""
import numpy as np
import pandas as pd
import pytest

from src.features.build import compute_features


def make_bars(n=2000, seed=0):
    """Synthetic 1-minute bars following a random walk."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({
        "open_time": np.arange(n) * 60000,
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "quote_volume": rng.uniform(1e5, 2e5, n),
        "trades": rng.integers(100, 200, n),
        "taker_buy_quote": rng.uniform(4e4, 1e5, n),
        "large_quote_volume": rng.uniform(0, 1e4, n),
        "large_trades": rng.integers(0, 5, n),
        "max_trade_quote": rng.uniform(1e3, 1e4, n),
    })


def test_no_lookahead():
    bars = make_bars()
    t = 1700
    before = compute_features(bars).drop(columns="target").loc[t * 60000]

    changed = bars.copy()
    changed.loc[changed.index > t, ["open", "high", "low", "close"]] *= 1.5
    after = compute_features(changed).drop(columns="target").loc[t * 60000]

    pd.testing.assert_series_equal(before, after)


def test_target_is_next_hour_vol():
    bars = make_bars()
    t = 1500
    r = np.diff(np.log(bars["close"].to_numpy()))
    expected = np.sqrt(np.sum(r[t:t + 60] ** 2))
    assert compute_features(bars, horizon=60)["target"].iloc[t] == pytest.approx(expected)


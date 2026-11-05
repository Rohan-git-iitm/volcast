"""Unit tests for feature engineering, on synthetic bars with known answers."""
import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES, compute_features, prepare_bars, realized_vol


def make_bars(closes) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open_time": np.arange(len(c)) * 60000, "open": c, "high": c, "low": c, "close": c,
        "volume": 1.0, "quote_volume": 100.0, "trades": 10, "taker_buy_base": 0.5, "taker_buy_quote": 50.0,
    })


def test_prepare_bars_fills_missing_minutes():
    out = prepare_bars(make_bars([100, 101, 102]).drop(index=1))
    assert len(out) == 3
    assert out["close"].iloc[1] == 100
    assert out["volume"].iloc[1] == 0


def test_realized_vol_matches_formula():
    rv = realized_vol(pd.Series(np.full(120, 0.001**2)), 60)
    assert rv.iloc[-1] == pytest.approx(0.001 * np.sqrt(60))


def test_constant_price_has_zero_volatility():
    f = compute_features(make_bars(np.full(1600, 100.0)))
    assert (f["rv_1h"].dropna() == 0).all()


def test_target_is_next_hour_volatility():
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, 1600)))
    f = compute_features(make_bars(closes))
    r = np.diff(np.log(closes))
    i = 1500
    assert f["target"].iloc[i] == pytest.approx(np.sqrt(np.sum(r[i:i + 60] ** 2)))


def test_all_model_features_present():
    f = compute_features(make_bars(np.linspace(100, 110, 1600)))
    assert set(FEATURES) <= set(f.columns)
    assert f[FEATURES].iloc[-1].notna().all()

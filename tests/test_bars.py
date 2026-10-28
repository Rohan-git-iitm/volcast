"""Check Spark-built bars against Binance's official 1-minute candles."""
import pandas as pd
import pytest

from src.utils.config import ROOT

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
FIXTURE = ROOT / "tests" / "fixtures" / "BTCUSDT-1m-2026-07-01.zip"
BARS = ROOT / "data" / "bars" / "symbol=BTCUSDT" / "date=2026-07-01"


@pytest.fixture
def klines():
    df = pd.read_csv(FIXTURE, header=None, names=KLINE_COLS)
    df["open_time"] = df["open_time"].where(df["open_time"] < 1e14, df["open_time"] // 1000)
    return df.set_index("open_time").sort_index()


@pytest.fixture
def bars():
    return pd.read_parquet(BARS).set_index("open_time").sort_index()


def test_same_minutes(bars, klines):
    assert bars.index.equals(klines.index)


@pytest.mark.parametrize("col", ["open", "high", "low", "close", "trades"])
def test_exact_columns(bars, klines, col):
    assert (bars[col] == klines[col]).all()


@pytest.mark.parametrize("col", ["volume", "quote_volume", "taker_buy_base", "taker_buy_quote"])
def test_summed_columns(bars, klines, col):
    assert bars[col].values == pytest.approx(klines[col].values, rel=1e-6)


def test_large_trade_columns(bars):
    assert (bars["large_trades"] <= bars["trades"]).all()
    assert (bars["large_quote_volume"] <= bars["quote_volume"] + 1e-6).all()
    assert (bars["max_trade_quote"] <= bars["quote_volume"] + 1e-6).all()
    assert (bars["large_trades"] > 0).any()

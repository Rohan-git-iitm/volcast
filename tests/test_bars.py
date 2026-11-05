"""Check Spark-built bars for the last day in the window against Binance's official 1-minute candles."""
import pandas as pd
import pytest
import requests

from src.utils.config import ROOT, load_config

pytestmark = pytest.mark.data

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
DAY = load_config()["data"]["end_date"]
BARS = ROOT / "data" / "bars" / "symbol=BTCUSDT" / f"date={DAY}"
CACHE = ROOT / "tests" / "fixtures" / "klines" / f"BTCUSDT-1m-{DAY}.zip"
URL = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{DAY}.zip"


@pytest.fixture(scope="module")
def klines():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        CACHE.write_bytes(resp.content)
    df = pd.read_csv(CACHE, header=None, names=KLINE_COLS)
    df["open_time"] = df["open_time"].where(df["open_time"] < 1e14, df["open_time"] // 1000)
    return df.set_index("open_time").sort_index()


@pytest.fixture(scope="module")
def bars():
    if not BARS.exists():
        pytest.skip(f"No bars for {DAY}; run the pipeline first")
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

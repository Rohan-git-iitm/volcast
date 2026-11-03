"""Live forecast API: keeps a rolling candle window per coin, forecasts every minute, scores past forecasts."""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

from src.api import store
from src.features.build import compute_features
from src.models.train import MODEL_FEATURES
from src.utils.config import load_config

BINANCE = "https://data-api.binance.vision/api/v3"
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5001")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
LOOKBACK = 1500  # minutes; the 24-hour features need 1,441 candles

cfg = load_config()
SYMBOLS = cfg["data"]["symbols"]
HORIZON = cfg["model"]["horizon_minutes"]
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("volcast.api")
windows: dict[str, pd.DataFrame] = {}


def fetch_candles(symbol: str, start: int) -> pd.DataFrame:
    """Fetch completed 1-minute candles from `start` (ms) up to the current minute."""
    end = int(time.time() // 60 * 60000)  # start of the current, unfinished minute
    rows = []
    while start < end:
        resp = requests.get(f"{BINANCE}/klines", timeout=10, params={
            "symbol": symbol, "interval": "1m", "startTime": start, "endTime": end - 1, "limit": 1000})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 60000
    df = pd.DataFrame(rows, columns=KLINE_COLS).drop(columns=["close_time", "ignore"])
    return df.astype(float).astype({"open_time": "int64", "trades": "int64"})


def update_window(symbol: str) -> bool:
    """Add new candles to the coin's rolling window. Returns True if any arrived."""
    old = windows.get(symbol)
    if old is None:
        start = int(time.time() // 60 * 60000) - LOOKBACK * 60000
    else:
        start = int(old["open_time"].max()) + 60000
    new = fetch_candles(symbol, start)
    if new.empty:
        return False
    combined = new if old is None else pd.concat([old, new])
    windows[symbol] = combined.drop_duplicates("open_time").tail(LOOKBACK).reset_index(drop=True)
    return True


def forecast(symbol: str, feats: pd.DataFrame):
    """Forecast from the latest feature row and store it."""
    row = feats.iloc[[-1]].assign(symbol_id=SYMBOLS.index(symbol))[MODEL_FEATURES]
    if row.isna().any(axis=None):
        log.warning("Skipping %s: incomplete features", symbol)
        return
    payload = {"dataframe_split": row.to_dict(orient="split", index=False)}
    resp = requests.post(f"{MODEL_URL}/invocations", json=payload, timeout=10)
    resp.raise_for_status()
    store.add_forecast(symbol, int(row.index[0]), resp.json()["predictions"][0])


def score(symbol: str, feats: pd.DataFrame):
    """Fill in the actual volatility for past forecasts whose hour has fully passed."""
    actual = feats["target"].dropna()
    for as_of in store.pending(symbol):
        if as_of in actual.index:
            store.set_actual(symbol, as_of, float(actual[as_of]))


def run_once():
    for symbol in SYMBOLS:
        try:
            if update_window(symbol):
                feats = compute_features(windows[symbol], HORIZON)
                forecast(symbol, feats)
                score(symbol, feats)
        except Exception:
            log.exception("Update failed for %s", symbol)


def loop():
    while True:
        run_once()
        time.sleep(60 - time.time() % 60 + 3)  # wake 3 s into each new minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    threading.Thread(target=loop, daemon=True).start()
    yield


app = FastAPI(title="VolCast API", lifespan=lifespan)


def check_symbol(symbol: str):
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Unknown symbol. Choose from {SYMBOLS}")


def describe(row: dict) -> dict:
    return {
        "symbol": row["symbol"],
        "as_of": pd.to_datetime(row["as_of"], unit="ms", utc=True).isoformat(),
        "horizon_minutes": HORIZON,
        "forecast_vol": row["forecast"],
        "expected_move_pct": round(row["forecast"] * 100, 3),
        "actual_vol": row["actual"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    checks = {}
    for name, url in [("model_server", f"{MODEL_URL}/ping"), ("binance", f"{BINANCE}/ping")]:
        try:
            checks[name] = requests.get(url, timeout=5).ok
        except requests.RequestException:
            checks[name] = False
    if not all(checks.values()):
        raise HTTPException(503, checks)
    return checks


@app.get("/predict")
def predict(symbol: str = "BTCUSDT"):
    check_symbol(symbol)
    row = store.latest(symbol)
    if row is None:
        raise HTTPException(503, "Warming up: no forecast yet")
    return describe(row)


@app.get("/history")
def history(symbol: str = "BTCUSDT", hours: int = 24):
    check_symbol(symbol)
    since = int(time.time() * 1000) - hours * 3600 * 1000
    return [describe(r) for r in store.history(symbol, since)]

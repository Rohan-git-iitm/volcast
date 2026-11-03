"""Live forecast API: fetches Binance candles, computes features, queries the model server."""
import os
import time

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

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
app = FastAPI(title="VolCast API")


def fetch_candles(symbol: str, minutes: int = LOOKBACK) -> pd.DataFrame:
    """Fetch the last `minutes` completed 1-minute candles from Binance."""
    end = int(time.time() // 60 * 60000)  # start of the current, unfinished minute
    start, rows = end - minutes * 60000, []
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
    if symbol not in SYMBOLS:
        raise HTTPException(404, f"Unknown symbol. Choose from {SYMBOLS}")
    feats = compute_features(fetch_candles(symbol), HORIZON)
    row = feats.iloc[[-1]].assign(symbol_id=SYMBOLS.index(symbol))[MODEL_FEATURES]
    if row.isna().any(axis=None):
        raise HTTPException(503, "Not enough candle history to compute features")

    payload = {"dataframe_split": row.to_dict(orient="split", index=False)}
    resp = requests.post(f"{MODEL_URL}/invocations", json=payload, timeout=10)
    resp.raise_for_status()
    vol = resp.json()["predictions"][0]
    return {
        "symbol": symbol,
        "as_of": pd.to_datetime(row.index[0], unit="ms", utc=True).isoformat(),
        "horizon_minutes": HORIZON,
        "forecast_vol": vol,
        "expected_move_pct": round(vol * 100, 3),
    }

"""Live forecast API: keeps a rolling candle window per coin, forecasts every minute, scores past forecasts."""
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api import store
from src.features.build import compute_features
from src.models.train import MODEL_FEATURES, scores
from src.utils.config import ROOT, load_config

BINANCE = "https://data-api.binance.vision/api/v3"
MODEL_URL = os.getenv("MODEL_URL", "http://127.0.0.1:5001")
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
LOOKBACK = 1500  # minutes; the 24-hour features need 1,441 candles
ENDPOINTS = {"/predict", "/history", "/health", "/ready", "/metrics", "/admin/simulate-drift", "/accuracy", "/pipeline"}
LIVE_QLIKE = Gauge("volcast_live_qlike", "QLIKE of forecasts scored in the last 24 h", ["symbol"])

cfg = load_config()
SYMBOLS = cfg["data"]["symbols"]
HORIZON = cfg["model"]["horizon_minutes"]
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("volcast.api")
windows: dict[str, pd.DataFrame] = {}

REQUESTS = Counter("volcast_requests_total", "API requests", ["endpoint", "status"])
REQUEST_SECONDS = Histogram("volcast_request_seconds", "API request latency", ["endpoint"])
MODEL_SECONDS = Histogram("volcast_model_seconds", "Model server call latency")
WORKER_ERRORS = Counter("volcast_worker_errors_total", "Failed worker updates", ["symbol"])
FORECAST = Gauge("volcast_forecast_vol", "Latest volatility forecast", ["symbol"])
LAST_CANDLE = Gauge("volcast_last_candle_timestamp", "Open time of the newest candle (unix s)", ["symbol"])
LIVE_RMSPE = Gauge("volcast_live_rmspe", "RMSPE of forecasts scored in the last 24 h", ["symbol"])
SCORED = Gauge("volcast_scored_forecasts", "Forecasts scored in the last 24 h", ["symbol"])
PERCENTILE = Gauge("volcast_feature_percentile", "Training percentile of the latest live value",
                   ["symbol", "feature"])
OUT_OF_RANGE = Gauge("volcast_feature_out_of_range",
                     "Share of the last 24 h of live values outside the training 1st-99th percentile",
                     ["symbol", "feature"])
REFERENCE = json.loads((ROOT / "models/reference.json").read_text())
DRIFT_FEATURES = list(REFERENCE[SYMBOLS[0]])
live_rows: dict[str, deque] = {s: deque(maxlen=1440) for s in SYMBOLS}


def update_drift(symbol: str, feats: pd.DataFrame):
    """Add the newest feature row to the coin's 24 h buffer and compare it with training."""
    rows = feats[DRIFT_FEATURES].dropna()
    buffer = live_rows[symbol]
    if buffer:
        rows = rows.tail(1)
    buffer.extend(rows.to_dict("records"))
    if not buffer:
        return
    live = pd.DataFrame(buffer)
    for f in DRIFT_FEATURES:
        q = np.array(REFERENCE[symbol][f])
        values = live[f].to_numpy()
        PERCENTILE.labels(symbol, f).set(float(np.interp(values[-1], q, np.arange(101))))
        OUT_OF_RANGE.labels(symbol, f).set(float(np.mean((values < q[1]) | (values > q[99]))))

SIM = {"factor": 1.0}
SIM_FACTOR = Gauge("volcast_drift_simulation_factor", "Volatility factor applied to drift monitoring (1 = off)")
SIM_FACTOR.set(1.0)


def simulate_volatility(bars: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale every price move by `factor`, for drift-simulation demos only."""
    b = bars.copy()
    ref = np.log(b["close"].iloc[0])
    for col in ["open", "high", "low", "close"]:
        b[col] = np.exp(ref + factor * (np.log(b[col]) - ref))
    return b

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
    """Add new candles to the coin's rolling window. Returns True if any arrived.

    The first call fetches an extra day so the drift buffer starts with a full 24 hours.
    """
    old = windows.get(symbol)
    if old is None:
        start = int(time.time() // 60 * 60000) - (LOOKBACK + 1440) * 60000
    else:
        start = int(old["open_time"].max()) + 60000
    new = fetch_candles(symbol, start)
    if new.empty:
        return False
    combined = new if old is None else pd.concat([old, new]).drop_duplicates("open_time").tail(LOOKBACK)
    windows[symbol] = combined.reset_index(drop=True)
    LAST_CANDLE.labels(symbol).set(windows[symbol]["open_time"].max() / 1000)
    return True


def forecast(symbol: str, feats: pd.DataFrame):
    """Forecast from the latest feature row and store it."""
    row = feats.iloc[[-1]].assign(symbol_id=SYMBOLS.index(symbol))[MODEL_FEATURES]
    if row.isna().any(axis=None):
        log.warning("Skipping %s: incomplete features", symbol)
        return
    payload = {"dataframe_split": row.to_dict(orient="split", index=False)}
    with MODEL_SECONDS.time():
        resp = requests.post(f"{MODEL_URL}/invocations", json=payload, timeout=10)
    resp.raise_for_status()
    vol = resp.json()["predictions"][0]
    store.add_forecast(symbol, int(row.index[0]), vol)
    FORECAST.labels(symbol).set(vol)


def score(symbol: str, feats: pd.DataFrame):
    """Fill in the actual volatility for past forecasts whose hour has fully passed."""
    actual = feats["target"].dropna()
    for as_of in store.pending(symbol):
        if as_of in actual.index:
            store.set_actual(symbol, as_of, float(actual[as_of]))


def update_live_error(symbol: str):
    """Recompute live QLIKE and RMSPE over forecasts scored in the last 24 hours, with training's scorer."""
    since = int(time.time() * 1000) - 24 * 3600 * 1000
    rows = [r for r in store.history(symbol, since) if r["actual"]]
    SCORED.labels(symbol).set(len(rows))
    if rows:
        s = scores(np.array([r["actual"] for r in rows]), np.array([r["forecast"] for r in rows]))
        LIVE_QLIKE.labels(symbol).set(s["qlike"])
        LIVE_RMSPE.labels(symbol).set(s["rmspe"])


def run_once():
    for symbol in SYMBOLS:
        try:
            if update_window(symbol):
                feats = compute_features(windows[symbol], HORIZON)
                forecast(symbol, feats)
                score(symbol, feats)
                update_live_error(symbol)
                if SIM["factor"] != 1:
                    feats = compute_features(simulate_volatility(windows[symbol], SIM["factor"]), HORIZON)
                update_drift(symbol, feats)
        except Exception:
            WORKER_ERRORS.labels(symbol).inc()
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


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path if request.url.path in ENDPOINTS else "other"
    REQUEST_SECONDS.labels(endpoint).observe(time.perf_counter() - start)
    REQUESTS.labels(endpoint, str(response.status_code)).inc()
    return response


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

BANDS = [(33, "Calm"), (67, "Normal"), (95, "Elevated"), (101, "Extreme")]


def risk_band(symbol: str, vol: float) -> tuple[float, str]:
    """Training percentile of a forecast among past hourly volatilities, and its band name."""
    pct = float(np.interp(vol, np.array(REFERENCE[symbol]["rv_1h"]), np.arange(101)))
    return pct, next(name for limit, name in BANDS if pct < limit)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    out = describe(row)
    pct, band = risk_band(symbol, row["forecast"])
    out.update(percentile=round(pct), band=band)
    window = windows.get(symbol)
    if window is not None:
        price = float(window["close"].iloc[-1])
        out.update(price=price, expected_move_usd=round(price * row["forecast"], 2))
    return out


@app.get("/accuracy")
def accuracy(symbol: str = "BTCUSDT", hours: int = 24):
    check_symbol(symbol)
    since = int(time.time() * 1000) - hours * 3600 * 1000
    rows = [r for r in store.history(symbol, since) if r["actual"]]
    if not rows:
        return {"symbol": symbol, "hours": hours, "scored": 0}
    s = scores(np.array([r["actual"] for r in rows]), np.array([r["forecast"] for r in rows]))
    return {"symbol": symbol, "hours": hours, "scored": len(rows),
            **{k: float(v) if np.isfinite(v) else None for k, v in s.items()}}


@app.get("/history")
def history(symbol: str = "BTCUSDT", hours: int = 24):
    check_symbol(symbol)
    since = int(time.time() * 1000) - hours * 3600 * 1000
    return [describe(r) for r in store.history(symbol, since)]

@app.post("/admin/simulate-drift")
def simulate_drift(factor: float = 3.0):
    if factor <= 0:
        raise HTTPException(400, "factor must be positive")
    SIM["factor"] = factor
    SIM_FACTOR.set(factor)
    for buffer in live_rows.values():
        buffer.clear()
    return {"factor": factor, "note": "Affects drift metrics only; forecasts use real data"}

PIPELINE_STATUS = Path(os.getenv("PIPELINE_STATUS", str(ROOT / "status/pipeline.json")))


@app.get("/pipeline")
def pipeline():
    try:
        return json.loads(PIPELINE_STATUS.read_text())
    except FileNotFoundError:
        raise HTTPException(503, "No pipeline status yet")
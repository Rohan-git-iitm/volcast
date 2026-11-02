"""Feature engineering shared by the training pipeline and the live API."""
import numpy as np
import pandas as pd

HOUR = 60


def prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Index bars by minute and fill any minutes that had no trades."""
    bars = bars.set_index("open_time").sort_index()
    full = pd.RangeIndex(bars.index.min(), bars.index.max() + 60000, 60000)
    bars = bars.reindex(full)
    bars["close"] = bars["close"].ffill()
    for col in ["open", "high", "low"]:
        bars[col] = bars[col].fillna(bars["close"])
    return bars.fillna(0)


def realized_vol(r2: pd.Series, window: int) -> pd.Series:
    """Realized volatility over the last `window` minutes, scaled to a 1-hour equivalent."""
    return np.sqrt(r2.rolling(window).sum() * HOUR / window)


def compute_features(bars: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    """Compute model features and the next-`horizon`-minutes volatility target."""
    b = prepare_bars(bars)
    r2 = np.log(b["close"]).diff() ** 2
    f = pd.DataFrame(index=b.index)

    for name, window in [("rv_15m", 15), ("rv_1h", 60), ("rv_4h", 240), ("rv_24h", 1440)]:
        f[name] = realized_vol(r2, window)
    f["range_1h"] = np.log(b["high"].rolling(60).max() / b["low"].rolling(60).min())

    vol_1h = b["quote_volume"].rolling(60).sum()
    f["volume_ratio"] = vol_1h / (b["quote_volume"].rolling(1440).sum() / 24)
    f["trades_ratio"] = b["trades"].rolling(60).sum() / (b["trades"].rolling(1440).sum() / 24)
    f["taker_buy_ratio"] = b["taker_buy_quote"].rolling(60).sum() / vol_1h


    t = pd.to_datetime(f.index, unit="ms", utc=True)
    f["hour"] = t.hour
    f["weekday"] = t.weekday

    f["target"] = np.sqrt(r2.rolling(horizon).sum().shift(-horizon))
    return f.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "rv_15m", "rv_1h", "rv_4h", "rv_24h", "range_1h",
    "volume_ratio", "trades_ratio", "taker_buy_ratio",
    "hour", "weekday",
]

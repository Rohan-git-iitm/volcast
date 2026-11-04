"""Features stage: build the training dataset from 1-minute bars inside the configured date window."""
import pandas as pd

from src.features.build import compute_features
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("features")


def window_ms(data_cfg: dict) -> tuple[int, int]:
    """Start of start_date and end of end_date (UTC), as millisecond timestamps."""
    start = pd.Timestamp(data_cfg["start_date"], tz="UTC")
    end = pd.Timestamp(data_cfg["end_date"], tz="UTC") + pd.Timedelta(days=1)
    return start.value // 10**6, end.value // 10**6


def main():
    cfg = load_config()
    bars_dir = ROOT / cfg["data"]["bars_dir"]
    out_dir = ROOT / cfg["data"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = window_ms(cfg["data"])

    frames = []
    for symbol in cfg["data"]["symbols"]:
        bars = pd.read_parquet(bars_dir / f"symbol={symbol}").drop(columns=["date"], errors="ignore")
        bars = bars[(bars["open_time"] >= start) & (bars["open_time"] < end)]
        f = compute_features(bars, cfg["model"]["horizon_minutes"]).dropna()
        f["symbol"] = symbol
        frames.append(f)
        logger.info(f"{symbol}: {len(f):,} rows")

    df = pd.concat(frames).rename_axis("open_time").reset_index()
    df.to_parquet(out_dir / "features.parquet", index=False)
    logger.info(f"Saved {len(df):,} rows with {df.shape[1]} columns")


if __name__ == "__main__":
    main()

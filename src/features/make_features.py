"""Features stage: build the training dataset from 1-minute bars."""
import pandas as pd

from src.features.build import compute_features
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("features")


def main():
    cfg = load_config()
    bars_dir = ROOT / cfg["data"]["bars_dir"]
    out_dir = ROOT / cfg["data"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for symbol in cfg["data"]["symbols"]:
        bars = pd.read_parquet(bars_dir / f"symbol={symbol}").drop(columns=["date"], errors="ignore")
        f = compute_features(bars, cfg["model"]["horizon_minutes"]).dropna()
        f["symbol"] = symbol
        frames.append(f)
        logger.info(f"{symbol}: {len(f):,} rows")

    df = pd.concat(frames).rename_axis("open_time").reset_index()
    df.to_parquet(out_dir / "features.parquet", index=False)
    logger.info(f"Saved {len(df):,} rows with {df.shape[1]} columns")


if __name__ == "__main__":
    main()

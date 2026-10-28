"""Thresholds stage: per-coin trade size above which a trade counts as large."""
import json
import tempfile

from pyspark.sql import functions as F

from src.data.trades import read_trades
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger
from src.utils.spark import get_spark

logger = get_logger("thresholds")


def main():
    cfg = load_config()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    quantile = cfg["data"]["large_trade_quantile"]
    n_days = cfg["data"]["threshold_days"]

    spark = get_spark(cfg)
    thresholds = {}
    try:
        for symbol in cfg["data"]["symbols"]:
            zips = sorted(raw_dir.glob(f"{symbol}-trades-*.zip"))[:n_days]
            with tempfile.TemporaryDirectory() as tmp:
                trades = read_trades(spark, zips, tmp)
                value = trades.agg(F.percentile_approx("quote_qty", quantile)).first()[0]
            thresholds[symbol] = float(value)
            logger.info(f"{symbol}: large trade >= {value:,.2f} USDT (from {len(zips)} days)")
    finally:
        spark.stop()

    out = ROOT / cfg["data"]["thresholds_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()

"""Bars stage: aggregate raw trades into 1-minute candles with Spark."""
import json
import tempfile
import time

from pyspark.sql import functions as F

from src.data.trades import read_trades
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger
from src.utils.spark import get_spark

logger = get_logger("bars")


def build_bars(trades, threshold: float):
    """Aggregate trades into 1-minute candles (Binance kline format plus large-trade columns)."""
    ts_ms = F.when(F.col("time") > 1e14, F.col("time") / 1000).otherwise(F.col("time")).cast("long")
    taker_buy = ~F.col("is_buyer_maker")
    large = F.col("quote_qty") >= threshold
    return (
        trades.withColumn("ts_ms", ts_ms)
        .withColumn("open_time", (F.col("ts_ms") / 60000).cast("long") * 60000)
        .groupBy("open_time")
        .agg(
            F.min_by("price", "trade_id").alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.max_by("price", "trade_id").alias("close"),
            F.sum("qty").alias("volume"),
            F.sum("quote_qty").alias("quote_volume"),
            F.count("*").alias("trades"),
            F.sum(F.when(taker_buy, F.col("qty")).otherwise(0.0)).alias("taker_buy_base"),
            F.sum(F.when(taker_buy, F.col("quote_qty")).otherwise(0.0)).alias("taker_buy_quote"),
            F.sum(F.when(large, F.col("quote_qty")).otherwise(0.0)).alias("large_quote_volume"),
            F.sum(F.when(large, 1).otherwise(0)).alias("large_trades"),
            F.max("quote_qty").alias("max_trade_quote"),
        )
        .withColumn("close_time", F.col("open_time") + 59999)
        .orderBy("open_time")
    )


def main():
    cfg = load_config()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    bars_dir = ROOT / cfg["data"]["bars_dir"]
    thresholds = json.loads((ROOT / cfg["data"]["thresholds_path"]).read_text())
    metrics_path = ROOT / "metrics" / "bars.json"
    metrics_path.parent.mkdir(exist_ok=True)

    spark = get_spark(cfg)
    total_trades, total_seconds, days, failed = 0, 0.0, 0, []
    try:
        for zip_path in sorted(raw_dir.glob("*.zip")):
            symbol, _, day = zip_path.stem.split("-", 2)
            out = bars_dir / f"symbol={symbol}" / f"date={day}"
            if (out / "_SUCCESS").exists():
                continue
            try:
                start = time.time()
                with tempfile.TemporaryDirectory() as tmp:
                    trades = read_trades(spark, [zip_path], tmp)
                    build_bars(trades, thresholds[symbol]).write.mode("overwrite").parquet(str(out))
                n = spark.read.parquet(str(out)).agg(F.sum("trades")).first()[0]
                seconds = time.time() - start
                total_trades, total_seconds, days = total_trades + n, total_seconds + seconds, days + 1
                logger.info(f"{symbol} {day}: {n:,} trades in {seconds:.1f}s ({n / seconds:,.0f} trades/s)")
            except Exception as e:
                logger.error(f"Failed {symbol} {day}: {e}")
                failed.append(f"{symbol} {day}")
    finally:
        spark.stop()

    metrics = {
        "days_processed": days,
        "trades_processed": total_trades,
        "seconds": round(total_seconds, 1),
        "trades_per_second": round(total_trades / total_seconds) if total_seconds else 0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Bars done: {metrics}")
    if failed:
        raise SystemExit(f"{len(failed)} days failed: {failed}")


if __name__ == "__main__":
    main()

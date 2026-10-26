"""Bars stage: aggregate raw trades into 1-minute candles with Spark."""
import json
import os
import tempfile
import time
import zipfile

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, LongType, StructField, StructType

from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("bars")

TRADE_SCHEMA = StructType([
    StructField("trade_id", LongType()),
    StructField("price", DoubleType()),
    StructField("qty", DoubleType()),
    StructField("quote_qty", DoubleType()),
    StructField("time", LongType()),
    StructField("is_buyer_maker", BooleanType()),
    StructField("is_best_match", BooleanType()),
])


def build_bars(trades):
    """Aggregate trades into 1-minute candles in Binance kline format."""
    ts_ms = F.when(F.col("time") > 1e14, F.col("time") / 1000).otherwise(F.col("time")).cast("long")
    taker_buy = ~F.col("is_buyer_maker")
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
        )
        .withColumn("close_time", F.col("open_time") + 59999)
        .orderBy("open_time")
    )


def main():
    cfg = load_config()
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    bars_dir = ROOT / cfg["data"]["bars_dir"]
    metrics_path = ROOT / "metrics" / "bars.json"
    metrics_path.parent.mkdir(exist_ok=True)

    os.environ["SPARK_LOCAL_IP"] = cfg["spark"]["local_ip"]
    spark = (
        SparkSession.builder.master(cfg["spark"]["master"])
        .config("spark.driver.memory", cfg["spark"]["driver_memory"])
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

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
                    zipfile.ZipFile(zip_path).extractall(tmp)
                    trades = spark.read.csv(tmp, schema=TRADE_SCHEMA, mode="DROPMALFORMED")
                    build_bars(trades).write.mode("overwrite").parquet(str(out))
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

"""Shared helpers for reading raw Binance trade archives with Spark."""
import zipfile

from pyspark.sql.types import BooleanType, DoubleType, LongType, StructField, StructType

TRADE_SCHEMA = StructType([
    StructField("trade_id", LongType()),
    StructField("price", DoubleType()),
    StructField("qty", DoubleType()),
    StructField("quote_qty", DoubleType()),
    StructField("time", LongType()),
    StructField("is_buyer_maker", BooleanType()),
    StructField("is_best_match", BooleanType()),
])


def read_trades(spark, zip_paths, tmp_dir):
    """Unzip archives into tmp_dir and return them as one Spark DataFrame."""
    for path in zip_paths:
        zipfile.ZipFile(path).extractall(tmp_dir)
    return spark.read.csv(str(tmp_dir), schema=TRADE_SCHEMA, mode="DROPMALFORMED")

"""Create a configured local Spark session."""
import os

from pyspark.sql import SparkSession


def get_spark(cfg: dict) -> SparkSession:
    """Start Spark in local mode with settings from the config."""
    os.environ["SPARK_LOCAL_IP"] = cfg["spark"]["local_ip"]
    spark = (
        SparkSession.builder.master(cfg["spark"]["master"])
        .config("spark.driver.memory", cfg["spark"]["driver_memory"])
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
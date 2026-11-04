"""Save the training feature percentiles per coin for live drift checks.

The reference describes the champion's training data, so it is rebuilt only when evaluate promotes a new champion.
"""
import json

import numpy as np
import pandas as pd

from src.features.build import FEATURES
from src.models.train import split
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("reference")
DRIFT_FEATURES = [f for f in FEATURES if f not in ("hour", "weekday")]
OUT = ROOT / "models/reference.json"


def main():
    decision = json.loads((ROOT / "metrics/evaluate.json").read_text())
    if OUT.exists() and not decision.get("promoted"):
        logger.info("Champion unchanged: keeping its drift reference")
        return
    m = load_config()["model"]
    df = pd.read_parquet(ROOT / "data/features/features.parquet")
    train, _ = split(df, m["test_days"], m["horizon_minutes"])
    ref = {}
    for symbol, g in train.groupby("symbol"):
        ref[symbol] = {f: np.quantile(g[f].dropna(), np.linspace(0, 1, 101)).tolist()
                       for f in DRIFT_FEATURES}
    OUT.write_text(json.dumps(ref, indent=1))
    logger.info("Built drift reference from the new champion's training data")


if __name__ == "__main__":
    main()

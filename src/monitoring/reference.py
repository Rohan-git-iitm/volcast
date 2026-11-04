"""Save the training feature percentiles per coin for live drift checks."""
import json

import numpy as np
import pandas as pd

from src.features.build import FEATURES
from src.models.train import split
from src.utils.config import ROOT, load_config

DRIFT_FEATURES = [f for f in FEATURES if f not in ("hour", "weekday")]


def main():
    m = load_config()["model"]
    df = pd.read_parquet(ROOT / "data/features/features.parquet")
    train, _ = split(df, m["test_days"], m["horizon_minutes"])
    ref = {}
    for symbol, g in train.groupby("symbol"):
        ref[symbol] = {f: np.quantile(g[f].dropna(), np.linspace(0, 1, 101)).tolist()
                       for f in DRIFT_FEATURES}
    (ROOT / "models/reference.json").write_text(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()

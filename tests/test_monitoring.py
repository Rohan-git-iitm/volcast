"""Unit tests for the risk band, drift metrics, drift simulation and the shared scorer."""
import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from src.api import main
from src.models.train import scores


def test_risk_band_follows_training_percentiles():
    q = main.REFERENCE["BTCUSDT"]["rv_1h"]
    assert main.risk_band("BTCUSDT", q[10])[1] == "Calm"
    assert main.risk_band("BTCUSDT", q[50])[1] == "Normal"
    assert main.risk_band("BTCUSDT", q[80])[1] == "Elevated"
    assert main.risk_band("BTCUSDT", q[100] * 2)[1] == "Extreme"


def test_drift_metrics():
    main.live_rows["BTCUSDT"].clear()
    ref = main.REFERENCE["BTCUSDT"]
    feats = pd.DataFrame({f: [ref[f][100] * 10] * 30 + [ref[f][50]] * 30 for f in main.DRIFT_FEATURES})
    main.update_drift("BTCUSDT", feats)
    labels = {"symbol": "BTCUSDT", "feature": "rv_1h"}
    assert REGISTRY.get_sample_value("volcast_feature_out_of_range", labels) == pytest.approx(0.5)
    assert 45 <= REGISTRY.get_sample_value("volcast_feature_percentile", labels) <= 55
    main.live_rows["BTCUSDT"].clear()


def test_simulate_volatility_scales_returns():
    c = np.array([100, 101, 99, 102, 100], dtype=float)
    bars = pd.DataFrame({"open": c, "high": c, "low": c, "close": c})
    out = main.simulate_volatility(bars, 3)
    np.testing.assert_allclose(np.diff(np.log(out["close"])), 3 * np.diff(np.log(c)))


def test_perfect_forecast_scores_zero():
    y = np.array([0.003, 0.004, 0.005])
    s = scores(y, y)
    assert s["qlike"] == pytest.approx(0, abs=1e-12)
    assert s["rmspe"] == pytest.approx(0, abs=1e-12)

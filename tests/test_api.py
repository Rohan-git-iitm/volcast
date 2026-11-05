"""API tests: every endpoint with a throwaway database; the model server and Binance are mocked."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.api import main, store

NOW = int(time.time() * 1000)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", str(tmp_path / "test.db"))
    store.init()
    return TestClient(main.app)  # not used as a context manager, so the background worker never starts


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_symbol_is_404(client):
    assert client.get("/predict", params={"symbol": "DOGEUSDT"}).status_code == 404


def test_predict_before_any_forecast_is_503(client):
    assert client.get("/predict").status_code == 503


def test_predict_returns_latest_forecast(client):
    store.add_forecast("BTCUSDT", NOW - 120000, 0.003)
    store.add_forecast("BTCUSDT", NOW - 60000, 0.004)
    body = client.get("/predict", params={"symbol": "BTCUSDT"}).json()
    assert body["forecast_vol"] == 0.004
    assert body["expected_move_pct"] == 0.4
    assert body["band"] in {"Calm", "Normal", "Elevated", "Extreme"}


def test_history_respects_hours(client):
    store.add_forecast("BTCUSDT", NOW - 60000, 0.004)
    store.add_forecast("BTCUSDT", NOW - 48 * 3600 * 1000, 0.005)
    assert len(client.get("/history", params={"hours": 24}).json()) == 1
    assert len(client.get("/history", params={"hours": 72}).json()) == 2


def test_accuracy_counts_only_scored_forecasts(client):
    store.add_forecast("BTCUSDT", NOW - 2 * 3600 * 1000, 0.004)
    assert client.get("/accuracy").json()["scored"] == 0
    store.set_actual("BTCUSDT", NOW - 2 * 3600 * 1000, 0.004)
    body = client.get("/accuracy").json()
    assert body["scored"] == 1
    assert body["qlike"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("ok, code", [(True, 200), (False, 503)])
def test_ready_reflects_dependencies(client, monkeypatch, ok, code):
    monkeypatch.setattr(main.requests, "get", lambda url, timeout: type("R", (), {"ok": ok})())
    assert client.get("/ready").status_code == code


def test_pipeline_status(client, tmp_path, monkeypatch):
    path = tmp_path / "pipeline.json"
    monkeypatch.setattr(main, "PIPELINE_STATUS", path)
    assert client.get("/pipeline").status_code == 503
    path.write_text(json.dumps({"window": {"start": "2026-08-04", "end": "2026-11-03"}}))
    assert client.get("/pipeline").json()["window"]["end"] == "2026-11-03"


def test_metrics_endpoint(client):
    client.get("/health")
    assert "volcast_requests_total" in client.get("/metrics").text

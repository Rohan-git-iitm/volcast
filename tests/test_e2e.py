"""End-to-end tests against the running Docker Compose stack. Run with: pytest -m e2e"""
import time

import pytest
import requests

pytestmark = pytest.mark.e2e

BACKEND = "http://localhost:8000"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get(url: str, **params):
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def test_backend_ready():
    assert get(f"{BACKEND}/ready") == {"model_server": True, "binance": True}


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_fresh_forecast_for_each_coin(symbol):
    body = get(f"{BACKEND}/predict", symbol=symbol)
    as_of = time.mktime(time.strptime(body["as_of"][:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    assert time.time() - as_of < 5 * 60, "newest forecast is more than 5 minutes old"
    assert 0 < body["forecast_vol"] < 0.1
    assert body["band"] in {"Calm", "Normal", "Elevated", "Extreme"}


def test_history_is_recorded():
    assert len(get(f"{BACKEND}/history", symbol="BTCUSDT", hours=1)) > 0


def test_metrics_cover_every_coin():
    text = requests.get(f"{BACKEND}/metrics", timeout=10).text
    for symbol in SYMBOLS:
        assert f'volcast_forecast_vol{{symbol="{symbol}"}}' in text


def test_pipeline_status_served():
    assert set(get(f"{BACKEND}/pipeline")["stages"].values()) <= {"ok", "waiting", "failed"}


def test_frontend_up():
    assert requests.get("http://localhost:8501/_stcore/health", timeout=10).text == "ok"


def test_prometheus_scrapes_backend():
    targets = get("http://localhost:9090/api/v1/targets")["data"]["activeTargets"]
    assert [t["health"] for t in targets if t["labels"]["job"] == "volcast-backend"] == ["up"]


def test_alert_rules_loaded():
    groups = get("http://localhost:9090/api/v1/rules")["data"]["groups"]
    names = {r["name"] for g in groups for r in g["rules"]}
    assert names == {"BackendDown", "DataStale", "WorkerErrors", "InputDrift", "LiveErrorHigh"}


def test_alertmanager_up():
    assert requests.get("http://localhost:9093/-/ready", timeout=10).status_code == 200


def test_grafana_dashboard_provisioned():
    assert get("http://localhost:3000/api/dashboards/uid/volcast")["dashboard"]["title"] == "VolCast"


def test_model_server_not_exposed():
    with pytest.raises(requests.ConnectionError):
        requests.get("http://localhost:5001/ping", timeout=5)

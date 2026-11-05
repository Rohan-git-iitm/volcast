# VolCast: test plan, test cases and test report

## 1. Test plan

### 1.1 Scope

**In scope**

| Component | What is tested |
|---|---|
| Feature engineering (`src/features/build.py`) | gap filling, realized-volatility formula, target alignment, feature completeness |
| Data pipeline output (`data/bars`) | Spark-built 1-minute bars against Binance's official candles |
| Monitoring logic (`src/api/main.py`) | risk bands, drift percentile and out-of-range share, drift simulation |
| Scoring (`src/models/train.py`) | QLIKE and RMSPE on a known case |
| Backend API | every endpoint, normal and error responses |
| Whole system (Docker Compose) | live forecasts, monitoring, alerting, frontend, network boundaries |
| Operations | rolling-window refresh failure handling, scheduling, drift alert, retraining loop |
| Code quality | static analysis with `ruff` |

**Out of scope**

- Availability of Binance's public endpoints (external dependency; failures are handled and tested, not prevented)
- Visual layout of the frontend (reviewed manually with screenshots)
- Forecast accuracy beyond the promotion gate (covered by the evaluate stage and live QLIKE monitoring, not by pass/fail tests)

### 1.2 Approach

| Level | Tool | Isolation |
|---|---|---|
| Unit | pytest | synthetic data with known answers; no network, no services |
| API (component) | FastAPI `TestClient` | throwaway SQLite database per test; model server and Binance replaced by fakes; background worker not started |
| Data validation | pytest | compares pipeline output with an independent source (Binance klines), cached locally |
| End to end | pytest + requests | the running Docker Compose stack, reached only through published ports |
| Operational | manual, scripted commands | the real pipeline and stack, with failures and alerts induced on purpose |
| Static analysis | ruff | whole repository |

Tests are grouped by markers in `pytest.ini`: `data` (needs pipeline data on disk) and `e2e` (needs the stack running). A plain `pytest` run excludes `e2e`.

### 1.3 Environment

| Item | Version |
|---|---|
| Host | MacBook, Ubuntu ARM64 VM in UTM |
| Python | 3.12.14 (uv-managed venv) |
| pytest | 9.2.0 |
| ruff | 0.17.3 |
| Services | Docker Compose: model server, backend, frontend, Prometheus, Alertmanager, Grafana |

### 1.4 How to run

```bash
python -m pytest -v               # unit, API and data tests
python -m pytest -m e2e -v        # end-to-end, with `docker compose up -d` running
ruff check src tests scripts frontend
```

### 1.5 Entry and exit criteria

**Entry:** the pipeline has run at least once (`dvc repro features`), so the data tests have bars for the window's last day; for end-to-end tests, all containers report healthy in `docker compose ps`.

**Exit (acceptance criteria):**

| ID | Criterion |
|---|---|
| AC1 | 100% of unit, API and data tests pass |
| AC2 | 100% of end-to-end tests pass against the running stack |
| AC3 | `ruff check` reports no findings |
| AC4 | Each coin's newest forecast is less than 5 minutes old (live chain working) |
| AC5 | A model is served only if it beats the persistence baseline and the current champion on QLIKE, on the same test period |
| AC6 | API p95 latency stays under 100 ms in normal operation (measured: about 8 ms) |
| AC7 | A failed data download never deletes existing data |
| AC8 | Sustained input drift raises the `InputDrift` alert; a retrain alert runs the gated retraining pipeline |

## 2. Test cases

### 2.1 Feature engineering (`tests/test_build.py`, `tests/test_features.py`)

| ID | Test | Checks | Expected |
|---|---|---|---|
| TC-F1 | `test_prepare_bars_fills_missing_minutes` | a missing minute is filled | previous close carried forward, volume 0 |
| TC-F2 | `test_realized_vol_matches_formula` | `realized_vol` on 60 returns of 0.1% | equals $0.001\sqrt{60}$ |
| TC-F3 | `test_constant_price_has_zero_volatility` | flat price series | `rv_1h` is 0 everywhere |
| TC-F4 | `test_target_is_next_hour_volatility` | target at minute $t$ | equals $\sqrt{\sum r^2}$ over the next 60 returns, computed independently |
| TC-F5 | `test_all_model_features_present` | features after 1,600 minutes | every model feature exists and has a value |
| TC-F6 | `test_no_lookahead` | features at minute 1,700 after every later price is multiplied by 1.5 | identical to the unchanged series: no feature uses future data |
| TC-F7 | `test_target_is_next_hour_vol` | target at minute $t$ on a random walk with a high/low spread | equals $\sqrt{\sum r^2}$ over the next 60 returns |

### 2.2 Monitoring and scoring (`tests/test_monitoring.py`)

| ID | Test | Checks | Expected |
|---|---|---|---|
| TC-M1 | `test_risk_band_follows_training_percentiles` | forecasts at training 10th, 50th, 80th percentile and twice the maximum | Calm, Normal, Elevated, Extreme |
| TC-M2 | `test_drift_metrics` | 30 values beyond training plus 30 at the median | out-of-range share 0.5; latest value's percentile 45 to 55 |
| TC-M3 | `test_simulate_volatility_scales_returns` | drift simulation with factor 3 | every log return exactly 3 times larger |
| TC-M4 | `test_perfect_forecast_scores_zero` | forecast equal to actual | QLIKE 0, RMSPE 0 |

### 2.3 Backend API (`tests/test_api.py`)

| ID | Test | Checks | Expected |
|---|---|---|---|
| TC-A1 | `test_health` | `/health` | 200, `{"status": "ok"}` |
| TC-A2 | `test_unknown_symbol_is_404` | `/predict?symbol=DOGEUSDT` | 404 |
| TC-A3 | `test_predict_before_any_forecast_is_503` | `/predict` with an empty log | 503 "warming up" |
| TC-A4 | `test_predict_returns_latest_forecast` | two stored forecasts | the newer one, as 0.4%, with a valid band |
| TC-A5 | `test_history_respects_hours` | forecasts 1 minute and 48 hours old | 1 row for `hours=24`, 2 for `hours=72` |
| TC-A6 | `test_accuracy_counts_only_scored_forecasts` | before and after an actual is filled | `scored` 0, then 1 with QLIKE 0 |
| TC-A7 | `test_ready_reflects_dependencies[True-200]` | both dependencies up | 200 |
| TC-A8 | `test_ready_reflects_dependencies[False-503]` | a dependency down | 503 |
| TC-A9 | `test_pipeline_status` | `/pipeline` before and after a status file exists | 503, then the file's contents |
| TC-A10 | `test_metrics_endpoint` | `/metrics` after a request | contains `volcast_requests_total` |

### 2.4 Data validation (`tests/test_bars.py`)

Runs on the last day of the current window, against Binance's official 1-minute candles for that day.

| ID | Test | Checks | Expected |
|---|---|---|---|
| TC-D1 | `test_same_minutes` | set of minutes | identical to official candles |
| TC-D2 to D6 | `test_exact_columns[open, high, low, close, trades]` | copied and counted values | exactly equal |
| TC-D7 to D10 | `test_summed_columns[volume, quote_volume, taker_buy_base, taker_buy_quote]` | summed values | equal within 1 part in a million (floating-point summation order) |
| TC-D11 | `test_large_trade_columns` | large-trade columns | never exceed their totals; at least one large trade |

### 2.5 End to end (`tests/test_e2e.py`)

| ID | Test | Checks | Expected |
|---|---|---|---|
| TC-E1 | `test_backend_ready` | `/ready` | model server and Binance both reachable |
| TC-E2 to E4 | `test_fresh_forecast_for_each_coin[BTC, ETH, SOL]` | newest forecast per coin | under 5 minutes old, between 0 and 10%, valid band |
| TC-E5 | `test_history_is_recorded` | `/history` for the last hour | at least one row |
| TC-E6 | `test_metrics_cover_every_coin` | backend metrics | a forecast gauge for each coin |
| TC-E7 | `test_pipeline_status_served` | `/pipeline` | every stage in a valid state |
| TC-E8 | `test_frontend_up` | Streamlit health endpoint | `ok` |
| TC-E9 | `test_prometheus_scrapes_backend` | Prometheus targets | backend target `up` |
| TC-E10 | `test_alert_rules_loaded` | Prometheus rules | exactly the five VolCast rules |
| TC-E11 | `test_alertmanager_up` | Alertmanager readiness | 200 |
| TC-E12 | `test_grafana_dashboard_provisioned` | Grafana dashboard `volcast` | exists with title VolCast |
| TC-E13 | `test_model_server_not_exposed` | port 5001 from the VM | connection refused |

### 2.6 Operational tests (manual)

| ID | Test | Procedure | Expected |
|---|---|---|---|
| TC-O1 | Rolling window first run | `./scripts/refresh.sh` after the window had not moved for weeks | window moves to the 92 days ending yesterday; old days removed; 276 files (92 × 3) |
| TC-O2 | Idempotent refresh | run `refresh.sh` twice on the same day | second run skips every stage, "Nothing to commit" |
| TC-O3 | Failed download deletes nothing | set `end_date` to today (file not yet published), run `dvc repro download` | non-zero exit; file count unchanged |
| TC-O4 | Scheduler environment | run `refresh.sh` with cron's bare environment (`env -i`) | succeeds; Java found |
| TC-O5 | Missed schedule | Mac asleep through the 13:00 run | run happens on wake (systemd timer, `Persistent=true`) |
| TC-O6 | Input drift alert | `scripts/simulate_drift.sh 3`, `InputDrift` shortened to `for: 2m` | drift tiles red; alert inactive → pending → firing |
| TC-O7 | Retraining loop | post a fake `LiveErrorHigh` alert to Alertmanager | listener starts `retrain.sh`; pipeline trains and evaluates; gate decides; run committed |
| TC-O8 | Download performance | `time` a refresh with nothing new | completes in seconds (verified-file markers) |

## 3. Test report

**Date:** 5 November 2026. **Commit:** after "Add end-to-end tests against the running stack". **Environment:** as in 1.3.

### 3.1 Results

| Level | Tests | Passed | Failed |
|---|---|---|---|
| Unit: features | 7 | 7 | 0 |
| Unit: monitoring and scoring | 4 | 4 | 0 |
| API | 10 | 10 | 0 |
| Data validation | 11 | 11 | 0 |
| End to end | 13 | 13 | 0 |
| **Automated total** | **45** | **45** | **0** |
| Operational (manual) | 8 | 8 | 0 |
| Static analysis (`ruff`) | | no findings | |

### 3.2 Acceptance criteria

| ID | Status | Evidence |
|---|---|---|
| AC1 | Met | 32 of 32 unit, API and data tests pass |
| AC2 | Met | 13 of 13 end-to-end tests pass |
| AC3 | Met | `ruff check`: All checks passed |
| AC4 | Met | TC-E2 to E4: forecasts for all three coins under 5 minutes old |
| AC5 | Met | TC-O7: challenger QLIKE 0.259 against champion 0.254, both scored on the same test period; not promoted, serving unchanged |
| AC6 | Met | Grafana API p95 about 8 ms; model server p95 10 to 25 ms after warm-up |
| AC7 | Met | TC-O3: download failed with exit code 255; 276 files before and after |
| AC8 | Met | TC-O6 and TC-O7 |

### 3.3 Defects found by testing

| ID | Found by | Defect | Fix |
|---|---|---|---|
| DEF-1 | TC-A6 | `/accuracy` returned a server error when correlation was undefined (one scored forecast gives NaN, which JSON cannot encode). Would have broken the accuracy panel after every restart. | Non-finite metrics returned as `null`; frontend shows "—" |
| DEF-2 | test run after rolling-window change | Bars test pinned to 1 July failed once that day left the window | Test now targets the window's last day, with official candles cached per day |
| DEF-3 | TC-O3 review | Cleanup ran before confirming downloads, so a failed download could have deleted data | Cleanup moved after `dvc repro download` succeeds |
| DEF-4 | TC-O1 review | Features stage read every bar on disk, so the "3-month" model would have trained on all history | Date filter in the features stage; window dates added as DVC params |
| DEF-5 | TC-O5 | Cron silently skipped the refresh while the VM was asleep | Replaced with a systemd timer with `Persistent=true` |
| DEF-6 | TC-O8 | Every refresh re-verified all 276 checksums and reran the download stage twice (about 4 minutes) | Verified-file markers, atomic `.part` downloads, `dvc commit -f download` after cleanup |
| DEF-7 | TC-O7 review | A non-promoted retrain overwrote the drift reference, so it no longer matched the served champion | Reference rebuilt only when a new champion is promoted; restored from DVC cache |

`ruff` also reported 9 findings: 6 style updates applied automatically, and 3 deliberate choices annotated (`check=False` in the retrain listener; per-day `except Exception` in the download and bars stages, which log each failure and fail the stage at the end).

### 3.4 Known limitations

- End-to-end tests depend on live Binance data, so they can fail for external reasons (Binance unreachable) without a VolCast defect.
- A manual rollback moves the `champion` alias without rebuilding the drift reference.
- The `LiveErrorHigh` threshold (0.38) is fixed in the rule; it does not move automatically when a new champion is promoted.
- Bars keep the large-trade threshold current on the day they were built; unused by the model (experiment E1).

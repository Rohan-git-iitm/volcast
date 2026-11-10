# VolCast: low-level design

## 1. Repository layout

```
volcast/
├── configs/config.yaml          all pipeline and model parameters
├── dvc.yaml, dvc.lock           pipeline definition and recorded versions
├── MLproject, python_env.yaml   MLflow project entry points and environment
├── requirements.txt             full pinned environment (training)
├── requirements-serve.txt       serving packages (pinned through requirements.txt)
├── Dockerfile, .dockerignore    serving image (model server and backend)
├── docker-compose.yml           all six services
├── src/
│   ├── utils/      config.py, logger.py, spark.py
│   ├── data/       download.py, trades.py, thresholds.py, bars.py, roll_window.py
│   ├── features/   build.py, make_features.py
│   ├── models/     train.py, evaluate.py, rollback.py
│   ├── monitoring/ reference.py, pipeline_status.py
│   └── api/        main.py, store.py
├── frontend/       app.py, pages/1_Pipeline_console.py, Dockerfile, requirements.txt
├── monitoring/     prometheus.yml, alerts.yml, alertmanager.yml, grafana/
├── scripts/        refresh.sh, retrain.sh, retrain_listener.py, simulate_drift.sh
├── tests/          unit, API, data and end-to-end tests
└── docs/           architecture, HLD, LLD, testing, experiments, user manual
```

## 2. Configuration (`configs/config.yaml`)

| Key | Value | Used by |
|---|---|---|
| `data.symbols` | BTCUSDT, ETHUSDT, SOLUSDT | every stage, backend |
| `data.start_date`, `data.end_date` | rolling 92-day window, set by `roll_window` | download, features |
| `data.base_url` | Binance daily trades archive | download |
| `data.raw_dir`, `bars_dir`, `features_dir` | `data/raw`, `data/bars`, `data/features` | pipeline stages |
| `data.large_trade_quantile`, `threshold_days` | 0.99, 7 | thresholds |
| `spark.master`, `driver_memory` | `local[*]`, 3g | bars, thresholds |
| `model.horizon_minutes` | 60 | features, train, backend |
| `model.test_days` | 14 | train, evaluate, reference |
| `model.lgbm` | 500 trees, learning rate 0.03, 31 leaves, min 200 samples per leaf | train |
| `mlflow.tracking_uri`, `experiment` | `sqlite:///mlflow.db`, `volcast` | train, evaluate, status |

`src/utils/config.py` exposes `ROOT` (repository root) and `load_config(path="configs/config.yaml")`.

## 3. DVC pipeline

| Stage | Command | Inputs | Params | Outputs |
|---|---|---|---|---|
| download | `python -m src.data.download` | `download.py` | `data.symbols`, `start_date`, `end_date` | `data/raw` (persist) |
| thresholds | `python -m src.data.thresholds` | `thresholds.py`, `trades.py`, `spark.py`, `data/raw` | `large_trade_quantile`, `threshold_days` | `artifacts/thresholds.json` (Git) |
| bars | `python -m src.data.bars` | `bars.py`, `trades.py`, `spark.py`, `data/raw`, `artifacts/thresholds.json` | | `data/bars` (persist), `metrics/bars.json` |
| features | `python -m src.features.make_features` | `build.py`, `make_features.py`, `data/bars` | `model.horizon_minutes`, `data.start_date`, `data.end_date` | `data/features` |
| train | `python -m src.models.train` | `train.py`, `build.py`, `data/features` | `model.test_days`, `model.lgbm` | `artifacts/train_runs.json`, `metrics/train.json` |
| evaluate | `python -m src.models.evaluate` | `evaluate.py`, `metrics/train.json`, `artifacts/train_runs.json`, `data/features` | | `metrics/evaluate.json` |
| export | `mlflow artifacts download --artifact-uri models:/volcast@champion` | `metrics/evaluate.json` | | `models/champion` |
| reference | `python -m src.monitoring.reference` | `reference.py`, `features.parquet`, `metrics/evaluate.json` | | `models/reference.json` (persist) |

`persist: true` keeps an output when its stage reruns, which makes `download` and `bars` incremental. The DVC cache uses hard links, so workspace files and cached files share disk space.

## 4. Data modules (`src/data/`)

**`download.py`.** `download_day(base_url, symbol, day, out_dir)` handles one day's zip:
1. If `name.zip` and `name.zip.ok` exist and the marker holds the zip's current size, skip.
2. Fetch Binance's `.CHECKSUM` file for the day.
3. If the zip exists and its SHA-256 matches, mark it verified; otherwise download to `name.zip.part`, verify, and rename to `name.zip` (atomic).
4. Write `name.zip.ok` containing the size.

Days are downloaded in parallel threads. Failures are logged per day; the stage exits non-zero at the end with the list of failed days.

**`thresholds.py`.** For each coin, computes the quote-value threshold above which a trade counts as large: the 99th percentile over the first 7 days of raw trades (Spark). Output: `artifacts/thresholds.json`, e.g. `{"BTCUSDT": 6826.63, ...}`.

**`bars.py`.** For each raw zip without a finished bar partition, reads the trades with Spark and aggregates them into 1-minute bars, written to `data/bars/symbol=<S>/date=<D>/` with a `_SUCCESS` marker. Days already marked are skipped. Writes throughput to `metrics/bars.json` (`days_processed`, `trades_processed`, `seconds`, `trades_per_second`).

**Bar schema**

| Column | Type | Definition |
|---|---|---|
| `open_time` | int64 | minute start, ms since epoch (UTC) |
| `open`, `high`, `low`, `close` | float | first, max, min, last trade price |
| `volume`, `quote_volume` | float | sum of base quantity, sum of price × quantity |
| `trades` | int | number of trades |
| `taker_buy_base`, `taker_buy_quote` | float | volumes where the buyer was the taker |
| `large_trades`, `large_quote_volume`, `max_trade_quote` | int, float, float | large-trade activity (not used by the model) |

**`roll_window.py`.**
- `set`: writes the 92 days ending yesterday (UTC) to the config, through a temporary file and an atomic rename.
- `cleanup`: deletes raw zips, markers, partial downloads and bar partitions dated before `start_date`.

## 5. Features (`src/features/`)

**`build.py`** (shared by training and serving)

| Function | Purpose |
|---|---|
| `prepare_bars(bars)` | index by `open_time`, insert missing minutes (close carried forward, open/high/low set to close, volumes 0) |
| `realized_vol(r2, window)` | $\sqrt{\frac{60}{w}\sum_{\text{last } w} r^2}$, scaled to a 1-hour equivalent |
| `compute_features(bars, horizon=60)` | returns the feature table and `target`; infinities replaced by NaN |

**Features**, with $r_t = \ln P_t - \ln P_{t-1}$:

| Feature | Definition |
|---|---|
| `rv_15m`, `rv_1h`, `rv_4h`, `rv_24h` | `realized_vol` over 15, 60, 240, 1,440 minutes |
| `range_1h` | $\ln(\max \text{high} / \min \text{low})$ over 60 minutes |
| `rv_same_hour_yday` | realized volatility of the same hour one day earlier |
| `vol_ratio_15m_24h`, `vol_ratio_1h_24h` | `rv_15m / rv_24h`, `rv_1h / rv_24h` |
| `volume_ratio`, `trades_ratio` | last hour's quote volume (trade count) over the 24-hour hourly average |
| `taker_buy_ratio` | taker-buy quote volume over total quote volume, last hour |
| `hour`, `weekday` | UTC time of day and day of week |
| `symbol_id` | coin index in `data.symbols` (added in training and serving) |
| `target` | $\sqrt{\sum_{i=1}^{60} r_{t+i}^2}$, the next hour's realized volatility |

The longest lookback is 1,440 minutes, so a complete feature row needs 1,441 candles.

**`make_features.py`.** Reads each coin's bars, keeps minutes from the start of `start_date` to the end of `end_date`, computes features, drops incomplete rows, and writes `data/features/features.parquet` (one row per coin and minute).

## 6. Models (`src/models/`)

**`train.py`**
- `split(df, test_days, horizon)`: the last `test_days` form the test set; training ends far enough before the test start that no training target overlaps it.
- `scores(actual, forecast)`: returns `qlike`, `rmspe`, `rmse`, `corr`, with $\text{QLIKE} = \text{mean}\left(\frac{y}{\hat y} - \ln\frac{y}{\hat y} - 1\right)$.
- Three models per run, each an MLflow run with parameters, overall and per-coin metrics:

| Model | Definition |
|---|---|
| persistence | forecast = `rv_1h` |
| har | linear regression of $\ln$ target on $\ln$ of `rv_1h`, `rv_4h`, `rv_24h` |
| lightgbm | `LGBMRegressor`, gamma objective, on all features plus `symbol_id` (categorical) |

LightGBM runs also log a BTC prediction plot, feature importance, and the model in skops format with its input signature. `MODEL_FEATURES = FEATURES + ["symbol_id"]`. Outputs: `metrics/train.json`, `artifacts/train_runs.json` (run IDs).

**`evaluate.py`**
1. Pick the best candidate by test QLIKE.
2. Reject it if it does not beat persistence.
3. Register it as a new version of `volcast`.
4. Re-score the current `champion` on the same test set (`champion_qlike`).
5. Promote (move the `champion` alias, previous champion becomes `previous`) only if the candidate's QLIKE is lower.

**`rollback.py`.** Swaps the `champion` and `previous` aliases, undoing the last promotion.

Output `metrics/evaluate.json`: `candidate`, `candidate_qlike`, `persistence_qlike`, `champion_qlike`, `version`, `promoted`, `reason`.

## 7. Monitoring modules (`src/monitoring/`)

**`reference.py`.** If a reference exists and `evaluate.json` says not promoted, keeps it. Otherwise writes `models/reference.json`: for each coin and each feature except `hour` and `weekday`, the 101 training-split percentiles (0th to 100th).

**`pipeline_status.py`.** `python -m src.monitoring.pipeline_status <refresh|retrain|init> <ok|failed> [champion_start champion_end]` writes `status/pipeline.json` atomically:

```json
{
  "stages": {"download": "ok", "train": "waiting", "...": "..."},
  "window": {"start": "2026-08-10", "end": "2026-11-09"},
  "throughput": 1584249,
  "refresh": {"time": "2026-11-10T11:56:00+00:00", "result": "ok"},
  "retrain": {"time": "...", "result": "ok", "promoted": false, "reason": "does not beat current champion"},
  "champion": {"version": 1, "window": {"start": "2026-07-01", "end": "2026-09-30"}, "qlike": 0.254},
  "runs": [{"time": "...", "model": "lightgbm", "qlike": 0.259, "rmspe": 0.346, "champion": false}],
  "updated": "..."
}
```

Stage states come from `dvc status`: `waiting` if inputs changed, `ok` otherwise; after a failed run, the first waiting stage is `failed`.

## 8. Backend (`src/api/`)

### 8.1 Prediction log (`store.py`)

SQLite at `PRED_DB` (default `predictions.db`; `/data/predictions.db` in Docker).

```sql
CREATE TABLE predictions (
    symbol   TEXT,
    as_of    INTEGER,   -- minute the forecast was made, ms since epoch
    forecast REAL,      -- forecast volatility for the next 60 minutes
    actual   REAL,      -- realized volatility, filled in once the hour has passed
    PRIMARY KEY (symbol, as_of)
);
```

Functions: `init`, `add_forecast` (insert or ignore), `set_actual`, `pending`, `latest`, `history(symbol, since)`.

### 8.2 Worker (`main.py`)

Runs in a background thread started at application startup, waking 3 seconds after each minute. For each coin:

| Step | Function | Detail |
|---|---|---|
| 1 | `update_window` | first call fetches 2,940 minutes of candles; later calls fetch only candles after the newest held, then trim to 1,500 |
| 2 | `compute_features` | same function as training |
| 3 | `forecast` | latest complete row → model server `/invocations` (`dataframe_split`) → `add_forecast` |
| 4 | `score` | fills `actual` for pending forecasts whose target is now in the window |
| 5 | `update_live_error` | live QLIKE and RMSPE over forecasts scored in the last 24 hours |
| 6 | `update_drift` | adds the latest feature row to a 24-hour buffer; updates percentile and out-of-range metrics |

Failures for one coin are logged and counted, and the other coins continue. Candles come from `https://data-api.binance.vision/api/v3/klines` (1,000 per request), excluding the unfinished current minute.

### 8.3 API

| Endpoint | Parameters | Response |
|---|---|---|
| `GET /health` | | `{"status": "ok"}` |
| `GET /ready` | | `{"model_server": true, "binance": true}`, or 503 with the same object if either is false |
| `GET /predict` | `symbol` (default BTCUSDT) | latest forecast (below); 404 unknown symbol; 503 before the first forecast |
| `GET /history` | `symbol`, `hours` (default 24) | list of forecasts made in the last `hours`, oldest first |
| `GET /accuracy` | `symbol`, `hours` (default 24) | `{"symbol", "hours", "scored", "qlike", "rmspe", "rmse", "corr"}`; undefined metrics are `null` |
| `GET /pipeline` | | contents of `status/pipeline.json`; 503 if absent |
| `GET /metrics` | | Prometheus text format |
| `POST /admin/simulate-drift` | `factor` (default 3, 1 = off) | `{"factor", "note"}`; scales price moves seen by drift monitoring only |

Example `/predict` response:

```json
{
  "symbol": "BTCUSDT",
  "as_of": "2026-11-04T17:25:00+00:00",
  "horizon_minutes": 60,
  "forecast_vol": 0.00437,
  "expected_move_pct": 0.437,
  "actual_vol": null,
  "percentile": 82,
  "band": "Elevated",
  "price": 80472.1,
  "expected_move_usd": 351.66
}
```

Risk bands use the forecast's percentile among the champion's training `rv_1h` values: Calm below 33, Normal below 67, Elevated below 95, Extreme above.

### 8.4 Metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `volcast_requests_total` | counter | endpoint, status | API requests |
| `volcast_request_seconds` | histogram | endpoint | API latency |
| `volcast_model_seconds` | histogram | | model server call latency |
| `volcast_worker_errors_total` | counter | symbol | failed minute updates |
| `volcast_forecast_vol` | gauge | symbol | latest forecast |
| `volcast_last_candle_timestamp` | gauge | symbol | newest candle's open time (unix seconds) |
| `volcast_live_qlike`, `volcast_live_rmspe` | gauge | symbol | live accuracy, last 24 hours |
| `volcast_scored_forecasts` | gauge | symbol | forecasts scored in the last 24 hours |
| `volcast_feature_percentile` | gauge | symbol, feature | training percentile of the latest value |
| `volcast_feature_out_of_range` | gauge | symbol, feature | share of the last 24 hours outside training's 1st to 99th percentile |
| `volcast_drift_simulation_factor` | gauge | | 1 unless a simulation is running |

### 8.5 Environment variables

| Variable | Default | Docker value |
|---|---|---|
| `MODEL_URL` | `http://127.0.0.1:5001` | `http://model-server:5001` |
| `PRED_DB` | `<repo>/predictions.db` | `/data/predictions.db` |
| `PIPELINE_STATUS` | `<repo>/status/pipeline.json` | same path, read-only mount |

## 9. Frontend (`frontend/`)

| Page | Calls | Shows |
|---|---|---|
| `app.py` (user page) | `/predict`, `/history`, `/accuracy` | expected move (% and $), risk band, price range, training percentile, forecast vs actual chart (6, 24, 72 h, gaps where offline), accuracy |
| `pages/1_Pipeline_console.py` | `/pipeline` | stage diagram, data window and last refresh, champion and last retrain, recent runs, embedded Grafana |

The user page reruns its data section every 60 seconds. Variables: `BACKEND_URL`, `GRAFANA_URL`, `DISPLAY_TZ` (default Asia/Kolkata).

## 10. Monitoring configuration (`monitoring/`)

- `prometheus.yml`: scrapes `backend:8000/metrics` every 15 s, evaluates rules every 15 s, sends alerts to `alertmanager:9093`.
- `alerts.yml`: the five rules in HLD 4.11.
- `alertmanager.yml`: default receiver does nothing; alerts matching `action="retrain"` go to `http://host.docker.internal:9095/retrain`; `group_wait` 30 s, `repeat_interval` 24 h.
- `grafana/`: provisioned Prometheus data source (uid `prometheus`) and the `volcast` dashboard.

## 11. Scripts (`scripts/`)

| Script | Trigger | Steps |
|---|---|---|
| `refresh.sh` | systemd timer, daily 13:00 IST (`Persistent=true`) | lock → `roll_window set` → `dvc repro download` → `roll_window cleanup` → `dvc commit -f download` → `dvc repro features` → commit → `dvc gc -w -f` → status |
| `retrain.sh` | retrain listener | lock → `dvc repro` → if promoted, `docker compose build && up -d` → commit → status |
| `retrain_listener.py` | cron `@reboot` | HTTP server on 9095; starts `retrain.sh` for a firing `action=retrain` alert, never twice at once, at most once per 24 hours |
| `simulate_drift.sh <factor>` | manual, for demos | calls `/admin/simulate-drift` |

Both pipeline scripts take the same `flock` lock (`.pipeline.lock`), so they never overlap, and record a `failed` status through an `ERR` trap if any step fails.

## 12. Containers (`docker-compose.yml`)

| Service | Image | Command | Health check |
|---|---|---|---|
| model-server | `volcast` (built) | `mlflow models serve -m models/champion -p 5001` | `/ping` |
| backend | `volcast` | `uvicorn src.api.main:app --port 8000` | `/health`; starts after model server is healthy |
| frontend | built from `frontend/` | `streamlit run app.py` | |
| prometheus | `prom/prometheus:v2.53.0` | | |
| alertmanager | `prom/alertmanager:v0.27.0` | | |
| grafana | `grafana/grafana:11.1.0` | | |

The `volcast` image installs `requirements-serve.txt` with versions constrained by `requirements.txt`, then copies `configs/`, `src/`, `models/champion/` and `models/reference.json`. Named volumes: `predictions`, `prometheus-data`, `grafana-data`.

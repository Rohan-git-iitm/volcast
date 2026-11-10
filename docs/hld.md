# VolCast: high-level design

## 1. Problem and goal

Traders and risk managers need to know how much a price is likely to move in the near future, not only its direction. VolCast forecasts the **realized volatility of the next 60 minutes** for BTC, ETH and SOL against USDT, updated every minute from live Binance data, and presents it as an expected percentage and dollar move with a risk band.

The forecast target at minute $t$ is

$$\sigma_t = \sqrt{\sum_{i=1}^{60} r_{t+i}^2}, \qquad r_t = \ln P_t - \ln P_{t-1},$$

where $P_t$ is the 1-minute closing price. It measures the size of the coming hour's price movement, in either direction.

The project also has to demonstrate an MLOps lifecycle end to end, running entirely on a local machine: versioned data, tracked experiments, a model registry with gated promotion, model serving, monitoring with drift detection, alerting, and automatic retraining.

## 2. Requirements

**Functional**

| ID | Requirement |
|---|---|
| FR1 | Produce a fresh forecast for each coin every minute |
| FR2 | Score each forecast against the actual volatility once its hour has passed |
| FR3 | Show users the current forecast, its history against actuals, and live accuracy |
| FR4 | Show operators the pipeline state, data window, champion model and recent runs |
| FR5 | Retrain on recent data when live accuracy degrades, and serve the new model only if it is better |
| FR6 | Detect when live inputs move outside the range the model was trained on |

**Non-functional**

| ID | Requirement |
|---|---|
| NFR1 | Reproducible: code, data, parameters and environment versioned; one command rebuilds the pipeline |
| NFR2 | Local only: no cloud services; everything runs on one VM |
| NFR3 | Loosely coupled: model, backend and frontend replaceable independently, communicating over REST |
| NFR4 | Observable: latency, errors, freshness, accuracy and drift visible on a dashboard, with alerts |
| NFR5 | Safe under failure: a failed download, training run or dependency must never corrupt data or replace the served model |
| NFR6 | Fast: API p95 under 100 ms (measured about 8 ms) |

## 3. System overview

The system has a **training side** (DVC pipeline, MLflow, scheduled refresh and retraining scripts on the VM) and a **serving side** (six containers in Docker Compose). They connect through the exported champion model, which is copied into the serving image, and through the retraining trigger. See `architecture.md` for the diagram, components and ports.

## 4. Design decisions

### 4.1 Data: raw trades from Binance's archive, aggregated with Spark

**Decision.** Build 1-minute bars from raw tick-level trades (about 552 million trades for three months and three coins) with Spark, rather than downloading Binance's ready-made candles.

**Why.** Raw trades allow features candles cannot provide, such as large-trade activity, and give full control over aggregation. Spark processes the volume in about 6 minutes on the VM (about 1.6 million trades per second), and the bars stage is incremental, so the daily refresh processes only the new day.

**Evidence.** A validation test compares Spark's bars with Binance's official candles for the newest day: minutes, open, high, low, close and trade counts match exactly; volumes match to one part in a million.

**Consequence.** The large-trade features were later dropped (4.3), so live serving needs only standard candles from Binance's REST API. Spark remains the batch engine for the training data.

### 4.2 Target and horizon: next-hour realized volatility

**Decision.** Forecast realized volatility over the next 60 minutes, at 1-minute resolution.

**Why.** Realized volatility is directly measurable from the same data, so every forecast can be scored objectively an hour later (FR2). One hour is long enough to be useful and short enough that live accuracy accumulates quickly. The horizon is a config parameter, so other horizons can be compared later.

### 4.3 Features: recent volatility, activity and time, without large-trade features

**Decision.** 13 features plus a coin identifier: realized volatility over 15 minutes, 1, 4 and 24 hours; the 1-hour high-low range; volatility in the same hour yesterday; two short-to-long volatility ratios; volume and trade-count ratios; the taker-buy share; hour of day and weekday.

**Why.** Volatility clusters: recent volatility is the strongest predictor of the next hour's. Ratios capture regime changes, and the time features capture strong intraday and weekly patterns.

**Evidence (experiment E1).** Large-trade features, built from raw trades in both absolute and relative ("surge") forms, changed every test metric by less than 0.4%, so they were excluded. This also removed the need for a live trade collector.

**Training and serving parity.** One function, `compute_features`, is used by the training pipeline and the live backend, so the two can never compute features differently. A unit test proves no feature uses future data.

### 4.4 Model: LightGBM, compared against persistence and HAR

**Decision.** Gradient-boosted trees (LightGBM) trained with the **gamma objective**, with two baselines trained in every run: persistence (next hour equals the last hour) and HAR, the standard econometric baseline, fitted as a linear regression of log volatility on the logs of 1-hour, 4-hour and 24-hour volatility.

**Why.** LightGBM captures non-linear interactions between volatility, activity and time, trains in seconds, and predicts in milliseconds. The gamma deviance has the same form as QLIKE, $\frac{y}{\hat y} - \ln\frac{y}{\hat y} - 1$, so the model is trained directly on the metric it is selected and monitored by (4.5). The baselines make every result interpretable: the model has to beat simple, well-known benchmarks to be worth serving.

### 4.5 Evaluation metric: QLIKE

**Decision.** Select and promote models on QLIKE; report RMSPE and correlation as secondary metrics.

**Why.** QLIKE is the standard loss for comparing volatility forecasts: it ranks forecasts consistently even though true volatility is only observed with noise, and it penalizes under-forecasting more than over-forecasting, matching the cost of being unprepared for a large move.

**Consistency.** The same `scores` function is used in training, evaluation, the live accuracy endpoint and the monitoring metric, so test and live numbers are directly comparable.

### 4.6 Promotion: a gate on the same test period

**Decision.** A new model is promoted to `champion` only if it beats both persistence and the **current champion re-scored on the same test period** (the last 14 days of the window).

**Why.** Comparing a challenger's score with the champion's old score would grade them on different fortnights. In the first automatic retrain, the challenger (QLIKE 0.259) lost to the re-scored champion (0.254) and was correctly rejected: newer data does not guarantee a better model.

### 4.7 Data window: rolling 92 days, refreshed daily, retrained on demand

**Decision.** Train on the 92 days ending yesterday. A daily scheduled job keeps the data current up to features; training runs only when a retrain is triggered.

**Why.** A fixed-length window keeps training cost constant and the model focused on recent market behavior. Separating data refresh from training means a retrain starts immediately on current data, and the model changes only when monitoring shows it should.

**Safety.** Old data is deleted only after the new window has downloaded and passed checksums; downloads are written to temporary files and renamed when verified; the features stage filters by date, so no stray data can leak into training. The schedule uses a systemd timer that runs missed jobs when the VM wakes.

### 4.8 Serving: a separate MLflow model server behind a FastAPI backend

**Decision.** The champion runs in its own container (`mlflow models serve`); a FastAPI backend does everything else.

**Why.** Separation lets the model change without touching backend code, and the backend change without touching the model (NFR3). The model server is not published outside Docker, so all access goes through the backend.

**Backend design.** A background worker forecasts every minute and writes to a SQLite prediction log; the API reads from the log. User requests are instant and never call Binance or the model, and many users cost no more than one. The worker keeps a rolling 25-hour candle window and fetches only new candles each minute, repairing gaps automatically after downtime.

### 4.9 Packaging: portable images with the model baked in

**Decision.** The serving image contains the code, configuration, the exported champion and its drift reference. A retrain that promotes a new champion rebuilds the image.

**Why.** An image with its model inside is immutable: what is deployed is exactly known, and it runs without access to the training machine's registry. Rollback means restarting a previous image or re-exporting a previous version. The `MLproject` file and pinned `requirements.txt` reproduce the training environment.

### 4.10 Drift detection: training percentiles instead of PSI

**Decision.** For each coin and feature, monitor the training percentile of the latest value and the share of the last 24 hours outside the training 1st to 99th percentile (2% expected).

**Why (experiment E4).** PSI was implemented first and reported near-maximum drift on normal data. The features are rolling windows, so consecutive minutes are almost identical; PSI and similar two-sample tests assume independent rows and break under this autocorrelation. Range-based checks make no independence assumption and match how tree models fail: they cannot extrapolate beyond the training range. On the same day, the percentile method correctly showed no drift.

### 4.11 Monitoring and alerting: Prometheus, Grafana, Alertmanager

**Decision.** The backend exposes Prometheus metrics; five alert rules; Alertmanager routes only alerts labelled `action=retrain` to the retraining listener.

| Alert | Condition | Purpose |
|---|---|---|
| BackendDown | scrape fails for 1 minute | service availability |
| DataStale | newest candle older than 10 minutes, for 2 minutes | data pipeline health |
| WorkerErrors | more than 3 failed updates in 15 minutes | worker health |
| InputDrift | a feature's out-of-range share above 20%, for 2 hours | early warning |
| LiveErrorHigh | live QLIKE above 1.5 times test, with 6 hours of scored forecasts, for 6 hours | triggers retraining |

**Why the long durations.** Forecasts for overlapping hours share most of their data, so one unusual hour moves many errors together. Only a sustained rise is evidence that the model has degraded. The retrain listener also allows at most one retrain per 24 hours.

### 4.12 Frontend: Streamlit, REST-only

**Decision.** A user page and a pipeline console in one Streamlit container, reading only from the backend.

**Why.** The frontend computes nothing: risk bands, dollar moves and accuracy all come from the backend, so the page can never disagree with monitoring. The console's pipeline data comes from a status file written by the training scripts and served by the backend, so the frontend never needs DVC or MLflow.

## 5. Failure handling

| Failure | Behavior |
|---|---|
| Binance unreachable | worker logs and counts the error; last forecast stays served; `DataStale` fires after 10 minutes |
| Model server down | `/ready` returns 503; worker errors counted; `WorkerErrors` fires |
| Daily download fails | refresh stops before deleting anything; console shows the stage as failed; next run retries |
| Training fails | `retrain.sh` stops; champion and containers unchanged; console shows the failure |
| Challenger worse than champion | not promoted; serving unchanged; run recorded |
| VM asleep at refresh time | the timer runs the refresh on wake; the backend repairs its candle window on its next minute |

## 6. Limitations and future work

- **Single machine.** All services run on one VM; serving stops while the Mac sleeps.
- **Fixed alert threshold.** The `LiveErrorHigh` threshold is fixed in the rule; it should follow each new champion's test QLIKE.
- **Rollback and drift reference.** A manual rollback moves the champion alias without restoring the matching drift reference. Logging the reference as an MLflow artifact of each model would tie them together.
- **Model updates reach clients by rebuild.** A central MLflow server with runtime model loading would let deployments follow the champion alias automatically, at the cost of depending on that server.
- **Further experiments.** Horizon comparison (15, 30, 60 minutes), hyperparameter tuning, LSTM and GARCH comparisons, and formal `dvc exp run` ablations.

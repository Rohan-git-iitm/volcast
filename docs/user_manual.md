# VolCast user manual

VolCast forecasts how much the prices of Bitcoin, Ethereum and Solana are likely to move over the next hour, updated every minute. This manual covers installation, daily use, demonstrations, operations and troubleshooting.

## 1. Requirements

| Item | Version or amount |
|---|---|
| OS | Ubuntu 24.04 or later (tested on ARM64 in a UTM VM; x86 works the same) |
| Python | 3.12 (3.13+ is not supported by PySpark) |
| uv | any recent version, for the Python environment |
| Java | OpenJDK 17, for Spark |
| Docker | Engine with the Compose plugin; your user in the `docker` group |
| Git | any recent version |
| Disk | at least 20 GB free (about 6 GB of data, 5 GB of images, plus headroom) |
| RAM | 8 GB recommended (Spark uses 3 GB) |
| Network | access to `data.binance.vision` and `data-api.binance.vision` |

## 2. Installation

### 2.1 Get the code and environment

```bash
git clone https://github.com/Rohan-git-iitm/volcast.git
cd volcast
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
echo 'export MLFLOW_TRACKING_URI=sqlite:///mlflow.db' >> .venv/bin/activate
source .venv/bin/activate
```

The last two lines make every activation of the venv point MLflow at the project's registry.

### 2.2 Build the data and train the first model

```bash
python -m src.data.roll_window set
dvc repro
```

The first line sets the window to the 92 days ending yesterday. `dvc repro` then runs all eight stages: it downloads about 4 GB of trades from Binance, builds bars with Spark, computes features, trains the three models, promotes the first champion, and exports it with its drift reference. Expect 15 to 30 minutes, mostly download and Spark. Equivalently, `mlflow run . -e main --env-manager local` runs the same pipeline as an MLflow project.

Then record the first status for the console, passing the training window just used:

```bash
python -m src.monitoring.pipeline_status init ok $(grep start_date configs/config.yaml | cut -d"'" -f2) $(grep end_date configs/config.yaml | cut -d"'" -f2)
```

### 2.3 Start the application

```bash
docker compose up -d --build
docker compose ps
```

The first build takes a few minutes. All six services should be up within a minute, with the model server and backend marked healthy. The first forecasts appear about a minute after the backend starts.

### 2.4 Schedule the daily refresh

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/volcast-refresh.service << EOF
[Unit]
Description=VolCast daily data refresh

[Service]
Type=oneshot
ExecStart=/bin/bash -c '$PWD/scripts/refresh.sh >> $PWD/logs/refresh.log 2>&1'
EOF

cat > ~/.config/systemd/user/volcast-refresh.timer << 'EOF'
[Unit]
Description=Run the VolCast data refresh daily at 13:00

[Timer]
OnCalendar=*-*-* 13:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now volcast-refresh.timer
loginctl enable-linger "$USER"
```

The refresh runs daily at 13:00 local time, and on wake if the machine was asleep at that time.

### 2.5 Start the retraining listener

```bash
mkdir -p logs
nohup .venv/bin/python scripts/retrain_listener.py >> logs/retrain_listener.log 2>&1 &
(crontab -l 2>/dev/null; echo "@reboot cd $PWD && .venv/bin/python scripts/retrain_listener.py >> logs/retrain_listener.log 2>&1") | crontab -
```

## 3. Using VolCast

| Page | Address | For |
|---|---|---|
| VolCast | `localhost:8501` | everyone: current forecasts |
| Pipeline console | `localhost:8501`, sidebar | operators: pipeline and model status |
| Grafana | `localhost:3000` | operators: monitoring (view without login; admin / admin to edit) |
| Prometheus | `localhost:9090` | operators: raw metrics, alert states |
| Alertmanager | `localhost:9093` | operators: active alerts |

### 3.1 The forecast page

1. **Pick a coin** at the top.
2. **The large number** is the expected move over the next hour, in either direction: ±0.44% means a typical hour from now sees the price within about 0.44% of where it is. Below it, the same in dollars at the current price.
3. **The band** compares the forecast with hours in the model's training period: **Calm** (quieter than two thirds of them), **Normal**, **Elevated** (busier than two thirds), **Extreme** (busier than 95%).
4. **Range and percentile** on the right state the dollar range and how unusual the forecast is.
5. **Forecast vs actual** plots past forecasts against what happened. The actual line trails by an hour, since each forecast can be checked only once its hour has passed. Gaps are times the service was off.
6. **Accuracy, last 24 h**: QLIKE (main score, lower is better, about 0.25 on the test set), average percentage error, correlation, and how many forecasts have been checked.

The page refreshes itself every minute. VolCast forecasts the size of price moves, never their direction, and is not financial advice.

### 3.2 The pipeline console

- **Stage diagram:** green up to date, amber waiting for new inputs (for example, training after a refresh), red failed in the last run.
- **Data:** the training window, the last refresh and its result, and Spark's throughput.
- **Champion model:** its version, training window, QLIKE on the latest test period, and the last retrain's outcome.
- **Recent training runs:** the latest five runs with their scores; the champion's run is ticked.
- **Monitoring:** the Grafana dashboard.

### 3.3 The Grafana dashboard

The top row has six tiles: backend up, data lag, worst live QLIKE, worst input drift, API latency, worker errors. If all are green, nothing needs attention. Below: forecasts and live accuracy, per-feature drift for the coin picked in the **Coin** menu, and system metrics. Every panel's ⓘ icon explains what it shows and what normal looks like.

## 4. Demonstrations

### 4.1 Input drift alert

```bash
sed -i '/alert: InputDrift/,/for:/ s/for: 2h/for: 2m/' monitoring/alerts.yml
docker compose restart prometheus
./scripts/simulate_drift.sh 3
```

Within a minute, the drift tiles and bars turn red. At `localhost:9090/alerts`, `InputDrift` goes from pending to firing two minutes later, and appears in Alertmanager. Forecasts keep using real data throughout. To reset:

```bash
./scripts/simulate_drift.sh 1
sed -i '/alert: InputDrift/,/for:/ s/for: 2m/for: 2h/' monitoring/alerts.yml
docker compose restart prometheus
```

### 4.2 Automatic retraining

```bash
curl -s -X POST localhost:9093/api/v2/alerts -H "Content-Type: application/json" \
  -d '[{"labels": {"alertname": "LiveErrorHigh", "action": "retrain", "symbol": "BTCUSDT"}}]'
tail -f logs/retrain.log
```

After about 30 seconds, Alertmanager calls the listener, which runs a real retrain: training, evaluation against the champion, and a rebuild of the containers only if the new model wins. The listener allows one retrain per 24 hours; restart it to repeat the demo sooner.

## 5. Operations

| Task | Command |
|---|---|
| Start or stop the application | `docker compose up -d` / `docker compose down` (data in volumes is kept) |
| Status of services | `docker compose ps` |
| Backend logs | `docker compose logs backend -f` |
| Run the refresh now | `systemctl --user start volcast-refresh.service` |
| Next scheduled refresh | `systemctl --user list-timers volcast-refresh.timer` |
| Retrain now | `./scripts/retrain.sh` |
| Run tests | `python -m pytest -v`; with the stack running, `python -m pytest -m e2e -v` |
| Style check | `ruff check src tests scripts frontend` |
| Free disk from old cached data | `dvc gc -w -f` |

### 5.1 Rolling back to the previous model

```bash
python -c "from mlflow import MlflowClient; c = MlflowClient(); v = c.get_model_version_by_alias('volcast', 'previous').version; c.set_registered_model_alias('volcast', 'champion', v); print('champion ->', v)"
dvc repro -f export
docker compose up -d --build
```

This moves the `champion` alias back to the previous version, exports it, and rebuilds the serving containers. The drift reference is not restored automatically (a known limitation), so drift readings describe the newer model's training data until the next promotion.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No Actual line on the chart | forecasts less than an hour old | wait; each actual appears an hour after its forecast |
| Console: download failed | network error or Binance file not yet published | `grep ERROR logs/refresh.log | tail`; rerun the refresh |
| Errors mentioning `No space left on device` | disk full | `df -h ~`; `dvc gc -w -f`; `docker builder prune -f` |
| Refresh did not run at 13:00 | machine asleep | runs on wake; check with `systemctl --user list-timers volcast-refresh.timer` |
| `InputDrift` firing with no simulation | the market is outside the training range | expected behavior; consider retraining once live QLIKE confirms it |

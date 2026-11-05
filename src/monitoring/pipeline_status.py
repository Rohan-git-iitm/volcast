"""Write status/pipeline.json for the pipeline console: stage states, data window, champion and recent runs.

Usage: python -m src.monitoring.pipeline_status <refresh|retrain|init> <ok|failed> [champion_start champion_end]
"""
import json
import subprocess
import sys
from datetime import UTC, datetime

import mlflow
from mlflow import MlflowClient

from src.utils.config import ROOT, load_config

STAGES = ["download", "thresholds", "bars", "features", "train", "evaluate", "export", "reference"]
OUT = ROOT / "status" / "pipeline.json"


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def stage_states(failed: bool) -> dict:
    """Each stage is ok (up to date) or waiting (inputs changed); after a failed run, the first waiting one failed."""
    out = subprocess.run(["dvc", "status", "--json"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    changed = {name.split(":")[-1] for name in json.loads(out or "{}")}
    states = {s: "waiting" if s in changed else "ok" for s in STAGES}
    if failed:
        first = next((s for s in STAGES if states[s] == "waiting"), None)
        if first:
            states[first] = "failed"
    return states


def model_info(cfg: dict):
    """Champion version and the 5 most recent training runs that have metrics."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()
    champ = client.get_model_version_by_alias("volcast", "champion")
    exp = client.get_experiment_by_name(cfg["mlflow"]["experiment"])
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=30)
    rows = [{"time": datetime.fromtimestamp(r.info.start_time / 1000, UTC).isoformat(),
             "model": r.info.run_name, "qlike": r.data.metrics["qlike"], "rmspe": r.data.metrics.get("rmspe"),
             "champion": r.info.run_id == champ.run_id}
            for r in runs if "qlike" in r.data.metrics][:5]
    return int(champ.version), rows


def main(job: str, result: str, champion_window=None):
    cfg = load_config()
    now = datetime.now(UTC).isoformat()
    status = read_json(OUT, {})
    status["stages"] = stage_states(result == "failed")
    status["window"] = {"start": cfg["data"]["start_date"], "end": cfg["data"]["end_date"]}
    status["throughput"] = read_json(ROOT / "metrics/bars.json", {}).get("trades_per_second")
    if job in ("refresh", "retrain"):
        status[job] = {"time": now, "result": result}

    decision = read_json(ROOT / "metrics/evaluate.json", {})
    champ = status.get("champion", {})
    if job == "retrain" and result == "ok":
        status["retrain"].update(promoted=decision.get("promoted"), reason=decision.get("reason"))
        if decision.get("promoted"):
            champ["window"] = status["window"]
    if champion_window:
        champ["window"] = {"start": champion_window[0], "end": champion_window[1]}
    version, runs = model_info(cfg)
    champ["version"] = version
    champ["qlike"] = decision.get("candidate_qlike") if decision.get("promoted") else decision.get("champion_qlike")
    status.update(champion=champ, runs=runs, updated=now)

    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(OUT)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:5] or None)

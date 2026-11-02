"""Evaluate stage: register the best model and promote it if it beats the current champion."""
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.models.train import HAR_FEATURES, MODEL_FEATURES, scores, split
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")
MODEL_NAME = "volcast"
CANDIDATES = {"har": HAR_FEATURES, "lightgbm": MODEL_FEATURES}


def load_test(cfg):
    """Rebuild the test set exactly as the train stage did."""
    df = pd.read_parquet(ROOT / cfg["data"]["features_dir"] / "features.parquet")
    df["symbol_id"] = df["symbol"].map({s: i for i, s in enumerate(cfg["data"]["symbols"])})
    return split(df, cfg["model"]["test_days"], cfg["model"]["horizon_minutes"])[1]


def champion_qlike(client, test):
    """Re-score the current champion on this test set. Returns (version, qlike), or (None, None) if none exists."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        return None, None
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    pred = model.predict(test[CANDIDATES[version.tags["model_type"]]])
    return version, scores(test["target"].to_numpy(), pred)["qlike"]


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()

    results = json.loads((ROOT / "metrics" / "train.json").read_text())
    runs = json.loads((ROOT / "artifacts" / "train_runs.json").read_text())

    best = min(CANDIDATES, key=lambda name: results[name]["qlike"])
    best_q, base_q = results[best]["qlike"], results["persistence"]["qlike"]
    decision = {"candidate": best, "candidate_qlike": best_q, "persistence_qlike": base_q, "promoted": False}

    if best_q >= base_q:
        decision["reason"] = "candidate does not beat persistence"
    else:
        champ, champ_q = champion_qlike(client, load_test(cfg))
        version = mlflow.register_model(f"runs:/{runs[best]}/model", MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, version.version, "model_type", best)
        client.set_model_version_tag(MODEL_NAME, version.version, "qlike", f"{best_q:.4f}")
        decision.update({"version": int(version.version), "champion_qlike": champ_q})

        if champ is None or best_q < champ_q:
            if champ is not None:
                client.set_registered_model_alias(MODEL_NAME, "previous", champ.version)
            client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
            decision.update({"promoted": True, "reason": "beats current champion" if champ else "first champion"})
        else:
            decision["reason"] = "does not beat current champion"

    logger.info(f"Decision: {decision}")
    (ROOT / "metrics" / "evaluate.json").write_text(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

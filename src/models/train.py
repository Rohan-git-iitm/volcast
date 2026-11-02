"""Train stage: fit baselines and LightGBM, track everything in MLflow."""
import json
import subprocess

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features.build import FEATURES
from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger("train")
HAR_FEATURES = ["rv_1h", "rv_4h", "rv_24h"]
MODEL_FEATURES = FEATURES + ["symbol_id"]
TRUSTED_TYPES = ["collections.OrderedDict", "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMRegressor"]


def split(df, test_days, horizon):
    """Last `test_days` are the test set; drop `horizon` minutes before it to avoid target overlap."""
    test_start = df["open_time"].max() - test_days * 86_400_000
    train = df[df["open_time"] < test_start - horizon * 60_000]
    test = df[df["open_time"] >= test_start]
    return train, test


def scores(y, pred):
    """RMSPE, RMSE and correlation between actual and predicted volatility."""
    return {
        "rmspe": float(np.sqrt(np.mean(((y - pred) / y) ** 2))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "corr": float(np.corrcoef(y, pred)[0, 1]),
    }


def git_info():
    """Current commit hash, and whether code or config has uncommitted changes."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"], text=True)
    return commit, bool(dirty.strip())


def data_version():
    """Hash of the features dataset, as recorded by DVC."""
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())
    return lock["stages"]["features"]["outs"][0]["md5"]


def log_metrics(test, pred):
    """Log overall and per-coin test metrics to the active MLflow run; return overall."""
    overall = scores(test["target"].to_numpy(), pred)
    mlflow.log_metrics(overall)
    for symbol in test["symbol"].unique():
        mask = (test["symbol"] == symbol).to_numpy()
        mlflow.log_metric(f"rmspe_{symbol}", scores(test["target"].to_numpy()[mask], pred[mask])["rmspe"])
    return overall


def plot_predictions(test, pred, symbol="BTCUSDT", days=3):
    """Actual vs predicted volatility for the last few test days of one coin."""
    mask = (test["symbol"] == symbol).to_numpy()
    t = pd.to_datetime(test["open_time"][mask], unit="ms")
    last = (t > t.max() - pd.Timedelta(days=days)).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[last], test["target"][mask][last], label="Actual", color="black")
    ax.plot(t[last], pred[mask][last], label="Predicted", alpha=0.8)
    ax.set_title(f"{symbol} next-hour volatility")
    ax.legend()
    return fig


def main():
    cfg = load_config()
    mcfg = cfg["model"]
    symbols = cfg["data"]["symbols"]

    df = pd.read_parquet(ROOT / cfg["data"]["features_dir"] / "features.parquet")
    df["symbol_id"] = df["symbol"].map({s: i for i, s in enumerate(symbols)})
    train, test = split(df, mcfg["test_days"], mcfg["horizon_minutes"])
    logger.info(f"Train rows: {len(train):,}  Test rows: {len(test):,}")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    commit, dirty = git_info()
    tags = {"git_commit": commit, "git_dirty": str(dirty), "data_version": data_version()}
    params = {
        "horizon_minutes": mcfg["horizon_minutes"], "test_days": mcfg["test_days"],
        "train_rows": len(train), "test_rows": len(test),
        "test_start": str(pd.to_datetime(test["open_time"].min(), unit="ms")),
    }

    log_target = {"func": np.log, "inverse_func": np.exp, "check_inverse": False}
    models = {
        "har": (TransformedTargetRegressor(
            regressor=make_pipeline(FunctionTransformer(np.log), LinearRegression()), **log_target),
            HAR_FEATURES, {}),
        "lightgbm": (TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**mcfg["lgbm"], verbose=-1), **log_target),
            MODEL_FEATURES, {"categorical_feature": ["symbol_id"]}),
    }

    results, runs = {}, {}

    with mlflow.start_run(run_name="persistence", tags=tags) as run:
        mlflow.log_params(params)
        pred = test["rv_1h"].to_numpy()
        results["persistence"], runs["persistence"] = log_metrics(test, pred), run.info.run_id

    for name, (model, cols, fit_params) in models.items():
        with mlflow.start_run(run_name=name, tags=tags) as run:
            mlflow.log_params({**params, "features": ",".join(cols)})
            if name == "lightgbm":
                mlflow.log_params(mcfg["lgbm"])
            model.fit(train[cols], train["target"], **fit_params)
            pred = model.predict(test[cols])
            results[name], runs[name] = log_metrics(test, pred), run.info.run_id

            mlflow.log_figure(plot_predictions(test, pred), "predictions_btc.png")
            if name == "lightgbm":
                imp = pd.Series(model.regressor_.booster_.feature_importance("gain"), index=cols).sort_values()
                fig, ax = plt.subplots(figsize=(8, 5))
                imp.plot.barh(ax=ax, title="Feature importance (gain)")
                mlflow.log_figure(fig, "feature_importance.png")
            mlflow.sklearn.log_model(
                model, name="model",
                signature=infer_signature(test[cols], pred),
                input_example=test[cols].head(3),
                skops_trusted_types=TRUSTED_TYPES,
            )
        logger.info(f"{name}: {results[name]}")

    (ROOT / "metrics").mkdir(exist_ok=True)
    (ROOT / "metrics" / "train.json").write_text(json.dumps(results, indent=2))
    (ROOT / "artifacts" / "train_runs.json").write_text(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()

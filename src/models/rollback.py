"""Swap the champion and previous model versions (undo the last promotion)."""
import mlflow
from mlflow import MlflowClient

from src.models.evaluate import MODEL_NAME
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("rollback")


def main():
    mlflow.set_tracking_uri(load_config()["mlflow"]["tracking_uri"])
    client = MlflowClient()
    champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    prev = client.get_model_version_by_alias(MODEL_NAME, "previous")
    client.set_registered_model_alias(MODEL_NAME, "champion", prev.version)
    client.set_registered_model_alias(MODEL_NAME, "previous", champ.version)
    logger.info(f"Rolled back: champion is now version {prev.version} (was {champ.version})")


if __name__ == "__main__":
    main()

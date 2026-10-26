import logging

from src.utils.config import ROOT, load_config


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to the console and logs/volcast.log."""
    cfg = load_config()["logging"]
    log_dir = ROOT / cfg["dir"]
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(cfg["level"])
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / "volcast.log")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger

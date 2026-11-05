"""Rolling data window: `set` moves it to the 92 days ending yesterday (UTC); `cleanup` deletes data before it."""
import shutil
import sys
from datetime import datetime, timedelta, timezone

import yaml

from src.utils.config import ROOT
from src.utils.logger import get_logger

logger = get_logger("roll_window")
WINDOW_DAYS = 92
CONFIG = ROOT / "configs" / "config.yaml"


def set_window():
    cfg = yaml.safe_load(CONFIG.read_text())
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=WINDOW_DAYS - 1)
    cfg["data"]["start_date"], cfg["data"]["end_date"] = start.isoformat(), end.isoformat()
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tmp.replace(CONFIG)
    logger.info(f"Window: {start} to {end}")


def cleanup():
    cfg = yaml.safe_load(CONFIG.read_text())
    start = cfg["data"]["start_date"]
    removed = 0
    for path in (ROOT / cfg["data"]["raw_dir"]).glob("*-trades-*.zip*"):
        if path.name.split("-trades-")[1][:10] < start:
            path.unlink()
            removed += 1
    for path in (ROOT / cfg["data"]["bars_dir"]).glob("symbol=*/date=*"):
        if path.name.split("=")[1] < start:
            shutil.rmtree(path)
            removed += 1
    logger.info(f"Removed {removed} raw files and bar partitions before {start}")


if __name__ == "__main__":
    {"set": set_window, "cleanup": cleanup}[sys.argv[1]]()

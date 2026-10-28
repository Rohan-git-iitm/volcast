"""Download stage: fetch daily Binance trade archives and verify checksums."""
import hashlib
from datetime import date, timedelta
from pathlib import Path

import requests

from src.utils.config import ROOT, load_config
from src.utils.logger import get_logger

logger = get_logger("download")


def date_range(start: str, end: str):
    """Yield each date from start to end, inclusive."""
    day = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while day <= last:
        yield day
        day += timedelta(days=1)


def sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_day(base_url: str, symbol: str, day: date, out_dir: Path) -> bool:
    """Download one day's zip if missing or corrupt. Return True on success."""
    name = f"{symbol}-trades-{day}.zip"
    path = out_dir / name
    url = f"{base_url}/{symbol}/{name}"
    try:
        r = requests.get(url + ".CHECKSUM", timeout=30)
        r.raise_for_status()
        expected = r.text.split()[0]

        if path.exists() and sha256(path) == expected:
            logger.info(f"Skip {name} (already verified)")
            return True

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

        if sha256(path) != expected:
            path.unlink()
            raise ValueError("checksum mismatch")
        logger.info(f"Downloaded {name}")
        return True
    except Exception as e:
        logger.error(f"Failed {name}: {e}")
        return False


def main():
    cfg = load_config()["data"]
    out_dir = ROOT / cfg["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for symbol in cfg["symbols"]:
        for day in date_range(cfg["start_date"], cfg["end_date"]):
            if not download_day(cfg["base_url"], symbol, day, out_dir):
                failed.append(f"{symbol} {day}")

    if failed:
        raise SystemExit(f"{len(failed)} downloads failed: {failed}")
    logger.info("All downloads complete")


if __name__ == "__main__":
    main()

"""SQLite prediction log: every live forecast and, once its hour has passed, the actual volatility."""
import os
import sqlite3

from src.utils.config import ROOT

DB = os.getenv("PRED_DB", str(ROOT / "predictions.db"))


def connect():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            symbol TEXT, as_of INTEGER, forecast REAL, actual REAL,
            PRIMARY KEY (symbol, as_of))""")


def add_forecast(symbol, as_of, forecast):
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO predictions (symbol, as_of, forecast) VALUES (?, ?, ?)",
                    (symbol, as_of, forecast))


def set_actual(symbol, as_of, actual):
    with connect() as con:
        con.execute("UPDATE predictions SET actual = ? WHERE symbol = ? AND as_of = ?",
                    (actual, symbol, as_of))


def pending(symbol):
    with connect() as con:
        rows = con.execute("SELECT as_of FROM predictions WHERE symbol = ? AND actual IS NULL", (symbol,))
        return [r["as_of"] for r in rows]


def latest(symbol):
    with connect() as con:
        row = con.execute("SELECT * FROM predictions WHERE symbol = ? ORDER BY as_of DESC LIMIT 1",
                          (symbol,)).fetchone()
    return dict(row) if row else None


def history(symbol, since):
    with connect() as con:
        rows = con.execute("SELECT * FROM predictions WHERE symbol = ? AND as_of >= ? ORDER BY as_of",
                           (symbol, since))
        return [dict(r) for r in rows]

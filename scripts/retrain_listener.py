"""Receive Alertmanager webhooks and start a retrain for firing alerts labelled action=retrain."""
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COOLDOWN = 24 * 3600  # at most one retrain per day
state = {"running": False, "last_start": 0.0}
lock = threading.Lock()


def run_retrain():
    with open(ROOT / "logs" / "retrain.log", "a") as log:
        subprocess.run([str(ROOT / "scripts" / "retrain.sh")], stdout=log, stderr=subprocess.STDOUT)
    with lock:
        state["running"] = False


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        wanted = any(a["status"] == "firing" and a["labels"].get("action") == "retrain"
                     for a in body.get("alerts", []))
        with lock:
            start = wanted and not state["running"] and time.time() - state["last_start"] > COOLDOWN
            if start:
                state.update(running=True, last_start=time.time())
        if start:
            threading.Thread(target=run_retrain, daemon=True).start()
        print(f"{time.strftime('%F %T')} webhook received: retrain {'started' if start else 'skipped'}", flush=True)
        self.send_response(202)
        self.end_headers()


HTTPServer(("0.0.0.0", 9095), Handler).serve_forever()

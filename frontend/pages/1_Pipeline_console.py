"""Pipeline console: data, training and serving status."""
import os

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("BACKEND_URL", "http://localhost:8000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Asia/Kolkata")
TEXT = "#111111"
STAGES = ["download", "thresholds", "bars", "features", "train", "evaluate", "export", "reference"]
STATE_COLORS = {"ok": "#2e7d32", "waiting": "#ef6c00", "failed": "#c62828"}
STATE_TEXT = {"ok": "up to date", "waiting": "waiting", "failed": "failed"}

st.set_page_config(page_title="VolCast pipeline", page_icon="🛠️", layout="wide")
st.markdown(
    f"""<style>
    .stMarkdown p, .stMarkdown li {{ font-size: 1.3rem; color: {TEXT}; }}
    [data-testid="stTable"] td, [data-testid="stTable"] th {{ font-size: 1.15rem; color: {TEXT}; }}
    </style>""",
    unsafe_allow_html=True,
)


def when(ts) -> str:
    return pd.to_datetime(ts).tz_convert(DISPLAY_TZ).strftime("%d %b %H:%M %Z") if ts else "—"


def day(d: str) -> str:
    return pd.to_datetime(d).strftime("%d %b %Y")


def diagram(states: dict):
    boxes = [
        f'<div style="flex:1; text-align:center; background:{STATE_COLORS[states.get(s, "ok")]}; color:white; '
        f'padding:14px 4px; border-radius:8px; font-size:1.1rem; font-weight:600;">{s}'
        f'<div style="font-weight:400; font-size:0.95rem;">{STATE_TEXT[states.get(s, "ok")]}</div></div>'
        for s in STAGES
    ]
    arrow = '<div style="align-self:center; font-size:1.6rem; color:#111;">→</div>'
    st.markdown(f'<div style="display:flex; gap:6px;">{arrow.join(boxes)}</div>', unsafe_allow_html=True)


st.title("Pipeline console")
st.markdown("**Data, training and serving status**")

try:
    resp = requests.get(f"{API}/pipeline", timeout=10)
    resp.raise_for_status()
    status = resp.json()
except requests.RequestException:
    st.error("Pipeline status is unavailable. It appears after the first refresh or retrain run.")
    st.stop()

diagram(status.get("stages", {}))

data, model = st.columns(2, gap="large")
with data:
    st.subheader("Data")
    w = status["window"]
    refresh = status.get("refresh", {})
    st.markdown(f"**Window:** {day(w['start'])} to {day(w['end'])}")
    st.markdown(f"**Last refresh:** {when(refresh.get('time'))}, {refresh.get('result', '—')}")
    if status.get("throughput"):
        st.markdown(f"**Bars throughput:** {status['throughput'] / 1e6:.1f} M trades/s")

with model:
    st.subheader("Champion model")
    champ = status.get("champion", {})
    st.markdown(f"**Version {champ.get('version', '—')}**")
    if "window" in champ:
        st.markdown(f"**Trained on:** {day(champ['window']['start'])} to {day(champ['window']['end'])}")
    if champ.get("qlike") is not None:
        st.markdown(f"**Test QLIKE:** {champ['qlike']:.3f}")
    retrain = status.get("retrain")
    if retrain:
        outcome = "promoted" if retrain.get("promoted") else f"not promoted ({retrain.get('reason', '')})"
        st.markdown(f"**Last retrain:** {when(retrain['time'])}, {retrain['result']}, {outcome}")
    else:
        st.markdown("**Last retrain:** —")

st.subheader("Recent training runs")
runs = pd.DataFrame(status.get("runs", []))
if runs.empty:
    st.markdown("No training runs recorded yet.")
else:
    st.table(pd.DataFrame({
        "Date": runs["time"].map(when),
        "Model": runs["model"],
        "QLIKE": runs["qlike"].map("{:.3f}".format),
        "RMSPE": runs["rmspe"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—"),
        "Champion": runs["champion"].map({True: "✓", False: ""}),
    }))

st.subheader("Monitoring")
components.iframe(f"{GRAFANA_URL}/d/volcast?orgId=1&kiosk&theme=light&refresh=30s", height=1500, scrolling=True)

"""VolCast: expected crypto price moves for the next hour."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("BACKEND_URL", "http://localhost:8000")
DISPLAY_TZ = os.getenv("DISPLAY_TZ", "Asia/Kolkata")
COINS = {"Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT", "Solana": "SOLUSDT"}
BAND_COLORS = {"Calm": "#2e7d32", "Normal": "#1565c0", "Elevated": "#ef6c00", "Extreme": "#c62828"}
TEXT = "#111111"

st.set_page_config(page_title="VolCast", page_icon="📈", layout="wide")
st.markdown(
    f"""<style>
    .stMarkdown p, .stMarkdown li {{ font-size: 1.3rem; color: {TEXT}; }}
    button p {{ font-size: 1.15rem; }}
    [data-testid="stWidgetLabel"] p {{ font-size: 1.15rem; color: {TEXT}; }}
    [data-testid="stMetricLabel"] p {{ font-size: 1.2rem; color: {TEXT}; }}
    [data-testid="stMetricValue"] {{ font-size: 2.6rem; color: {TEXT}; }}
    </style>""",
    unsafe_allow_html=True,
)


def get(path: str, **params):
    resp = requests.get(f"{API}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def usd(x: float) -> str:
    """Dollars with cents for small amounts, whole dollars for large ones."""
    return f"${x:,.2f}" if x < 1000 else f"${x:,.0f}"


def forecast_card(pred: dict):
    color = BAND_COLORS[pred["band"]]
    dollars = f"±{usd(pred['expected_move_usd'])} at {usd(pred['price'])}" if "price" in pred else ""
    st.markdown(
        f"""<div style="border-left:8px solid {color}; padding:0.5rem 1.5rem; color:{TEXT};">
<div style="font-size:4.5rem; font-weight:700; line-height:1.1;">±{pred['expected_move_pct']:.2f}%</div>
<div style="font-size:1.5rem; margin:0.4rem 0 0.8rem;">{dollars}</div>
<span style="background:{color}; color:white; font-size:1.3rem; font-weight:600; padding:4px 16px; border-radius:16px;">{pred['band']}</span>
</div>""",
        unsafe_allow_html=True,
    )


def facts(pred: dict):
    updated = pd.to_datetime(pred["as_of"]).tz_convert(DISPLAY_TZ).strftime("%H:%M %Z")
    lines = []
    if "price" in pred:
        low, high = pred["price"] - pred["expected_move_usd"], pred["price"] + pred["expected_move_usd"]
        lines.append(f"**Range:** {usd(low)} to {usd(high)}".replace("$", "\\$"))
    lines.append(f"**Busier than {pred['percentile']}% of past hours**")
    lines.append(f"**Updated {updated}**")
    for line in lines:
        st.markdown(line)


def history_chart(hist: pd.DataFrame, hours: int):
    if hist.empty:
        st.markdown("No forecasts in this period yet.")
        return
    df = hist.assign(time=pd.to_datetime(hist["as_of"]).dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None))
    df = df.sort_values("time")
    df["segment"] = (df["time"].diff() > pd.Timedelta(minutes=2)).cumsum()
    df = df.assign(Forecast=df["forecast_vol"] * 100, Actual=df["actual_vol"] * 100)
    long = df.melt(["time", "segment"], ["Forecast", "Actual"], var_name="Series", value_name="Move (%)").dropna()
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M" if hours <= 24 else "%d %b %H:%M")),
            y=alt.Y("Move (%):Q", title="Hourly move (%)"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Forecast", "Actual"], range=["#1565c0", "#424242"]),
                            legend=alt.Legend(orient="top", title=None)),
            detail="segment:N",
        )
        .configure_axis(labelFontSize=15, titleFontSize=16, labelColor=TEXT, titleColor=TEXT)
        .configure_legend(labelFontSize=17, labelColor=TEXT, symbolStrokeWidth=4)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("Actual appears one hour after each forecast. Gaps: service offline.")


def accuracy_panel(acc: dict):
    if acc["scored"] == 0:
        st.markdown("No forecasts checked yet. Each one is checked an hour after it is made.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QLIKE", f"{acc['qlike']:.3f}", help="Lower is better. Test: 0.25.")
    c2.metric("Error", f"{acc['rmspe'] * 100:.0f}%", help="Average percentage error.")
    c3.metric("Correlation", f"{acc['corr']:.2f}", help="1 is perfect.")
    c4.metric("Checked", f"{acc['scored']:,}")


st.title("VolCast")
st.markdown("**Expected crypto price moves for the next hour**")
coin = st.segmented_control("Coin", list(COINS), default="Bitcoin") or "Bitcoin"


@st.fragment(run_every="60s")
def dashboard(coin: str):
    symbol = COINS[coin]
    try:
        pred = get("/predict", symbol=symbol)
    except requests.RequestException:
        st.error("The forecasting service is unavailable right now. Retrying every minute.")
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        forecast_card(pred)
    with right:
        facts(pred)

    st.subheader("Forecast vs actual")
    hours = st.segmented_control("Period", [6, 24, 72], default=24, format_func=lambda h: f"{h} h", key="hours") or 24
    history_chart(pd.DataFrame(get("/history", symbol=symbol, hours=hours)), hours)

    st.subheader("Accuracy, last 24 h")
    accuracy_panel(get("/accuracy", symbol=symbol))


dashboard(coin)

"""
Streamlit dashboard for Fraud-Spike Sentinel.
Run:
    streamlit run app/dashboard.py
"""
from pathlib import Path

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Fraud-Spike Sentinel", page_icon="\U0001F6E1\uFE0F", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
DATA_PATH = st.sidebar.text_input("Replay data file", value="data/paysim.csv")

DECISION_COLORS = {
    "ALLOW": "#22c55e",
    "MONITOR": "#3b82f6",
    "REVIEW": "#f59e0b",
    "BLOCK": "#ef4444",
}
DECISION_ORDER = ["ALLOW", "MONITOR", "REVIEW", "BLOCK"]

CUSTOM_CSS = """
<style>
div[data-testid="stMetric"] {
    background-color: #161b26;
    border: 1px solid #262d3d;
    border-radius: 12px;
    padding: 14px 14px 10px 14px;
}
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
.decision-badge {
    padding: 3px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    color: white;
    display: inline-block;
}
.badge-ALLOW   { background-color: #22c55e; }
.badge-MONITOR { background-color: #3b82f6; }
.badge-REVIEW  { background-color: #f59e0b; }
.badge-BLOCK   { background-color: #ef4444; }
.feed-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 4px; border-bottom: 1px solid #1f2633;
    font-size: 0.85rem;
}
.feed-account { color: #9aa4b2; font-family: monospace; }
.explanation-box {
    background-color: #12161f; border-left: 3px solid #6366f1;
    padding: 10px 14px; border-radius: 6px; font-size: 0.85rem;
    color: #cdd5e0; margin-top: 4px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def decision_badge(decision: str) -> str:
    d = decision if decision in DECISION_COLORS else "MONITOR"
    return f'<span class="decision-badge badge-{d}">{decision}</span>'


def load_metrics():
    p = Path("model/metrics.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


st.title("Fraud-Spike Sentinel")
st.caption("AI Risk Manager \u2014 real-time transaction scoring + account-level spike detection")

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "feed" not in st.session_state:
    st.session_state.feed = []
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

tab_live, tab_perf, tab_account = st.tabs(["Live Feed", "Model Performance", "Account Lookup"])

#LIVE FEED
with tab_live:
    def _start_streaming():
        st.session_state.streaming = True

    def _stop_streaming():
        st.session_state.streaming = False

    def _reset_feed():
        st.session_state.feed = []
        st.session_state.cursor = 0

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    interval = ctrl1.slider("Interval (sec)", 0.5, 5.0, 1.5, 0.5)
    ctrl2.button("\u25B6 Start", on_click=_start_streaming,
                 disabled=st.session_state.streaming, use_container_width=True)
    ctrl3.button("\u23F8 Stop", on_click=_stop_streaming,
                 disabled=not st.session_state.streaming, use_container_width=True)
    ctrl4.button("Reset feed", on_click=_reset_feed, use_container_width=True)

    run_every = interval if st.session_state.streaming else None

    @st.fragment(run_every=run_every)
    def live_feed_fragment():
        if st.session_state.streaming:
            try:
                df = pd.read_csv(DATA_PATH)
            except FileNotFoundError:
                st.error(f"Couldn't find {DATA_PATH}")
                df = None

            if df is not None and len(df) > 0:
                if st.session_state.cursor >= len(df):
                    st.session_state.cursor = 0
                row = df.iloc[st.session_state.cursor]
                st.session_state.cursor += 1

                payload = {
                    "nameOrig": row["nameOrig"], "step": int(row["step"]), "amount": float(row["amount"]),
                    "type": row["type"], "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]), "nameDest": row["nameDest"],
                    "oldbalanceDest": float(row["oldbalanceDest"]), "newbalanceDest": float(row["newbalanceDest"]),
                }
                try:
                    resp = requests.post(f"{API_URL}/score", json=payload, timeout=5)
                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state.feed.insert(0, {**payload, **result})
                    st.session_state.feed = st.session_state.feed[:300]
                except requests.RequestException as e:
                    st.error(f"API call failed \u2014 is uvicorn running? ({e})")

        feed = st.session_state.feed

        if not feed:
            st.info("Click **Start** to begin streaming transactions through the live API.")
            return

        latest = feed[0]
        st.markdown("**Latest transaction**")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
        c1.markdown(f"<span class='feed-account'>{latest.get('nameOrig','?')}</span>", unsafe_allow_html=True)
        c2.markdown(f"\u20B9{latest.get('amount', 0):,.2f}")
        c3.markdown(decision_badge(latest.get("decision", "?")), unsafe_allow_html=True)  
        c4.markdown(f"score `{latest.get('score', 0):.3f}`")
        if latest.get("explanation"):  
            st.markdown(f'<div class="explanation-box">{latest["explanation"]}</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2])

        with left:
            st.markdown("**Recent feed**")
            rows_html = "<div>"
            for r in feed[:15]:
                rows_html += (
                    f'<div class="feed-row">'
                    f'<span class="feed-account">{r.get("nameOrig","?")}</span>'
                    f'<span>\u20B9{r.get("amount",0):,.0f}</span>'
                    f'<span>{r.get("type","")}</span>'
                    f'<span>{decision_badge(r.get("decision","?"))}</span>'
                    f'<span style="color:#9aa4b2">score {r.get("score",0):.3f}</span>'
                    f'</div>'
                )
            rows_html += "</div>"
            st.markdown(rows_html, unsafe_allow_html=True)

        with right:
            st.markdown("**Decisions so far**")
            counts = pd.Series([r.get("decision", "?") for r in feed]).value_counts()
            counts = counts.reindex(DECISION_ORDER).fillna(0)
            fig = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index, color_discrete_map=DECISION_COLORS,
                labels={"x": "", "y": "count"},
            )
            fig.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    live_feed_fragment()

# PERFORMANCE
with tab_perf:
    metrics = load_metrics()
    if not metrics:
        st.warning("No model/metrics.json found \u2014 run training first.")
    else:
        cols = st.columns(6)
        cols[0].metric("Precision", f"{metrics['precision']:.3f}")
        cols[1].metric("Recall", f"{metrics['recall']:.3f}")
        cols[2].metric("F1", f"{metrics['f1']:.3f}")
        cols[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        cols[4].metric("Threshold", f"{metrics['threshold']:.3f}")
        saved = metrics["cost_naive_half"] - metrics["cost_at_threshold"]
        cols[5].metric("Cost saved vs 0.5", f"{saved:,.0f}")

        st.divider()
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Cost vs. threshold**")
            sweep = pd.DataFrame(metrics["threshold_sweep"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["cost"], mode="lines", line=dict(color="#6366f1", width=2)))
            fig.add_vline(x=metrics["threshold"], line_dash="dash", line_color="#ef4444",
                          annotation_text="chosen", annotation_position="top")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              xaxis_title="threshold", yaxis_title="total cost")
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            st.markdown("**Top features**")
            imp = pd.DataFrame(metrics["top_features"][:8]).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                        color_discrete_sequence=["#6366f1"])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        if st.session_state.feed:
            st.markdown("**Live score distribution**")
            feed_df = pd.DataFrame(st.session_state.feed)
            fig = px.histogram(feed_df, x="score", color="decision", nbins=30,
                               color_discrete_map=DECISION_COLORS)
            fig.add_vline(x=metrics["threshold"], line_dash="dash", line_color="white")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Run the live feed to populate a real score distribution here.")

# ACCOUNT LOOKUP
with tab_account:
    name_orig = st.text_input("Account (nameOrig)", value="")
    if st.button("Check account") and name_orig:
        try:
            resp = requests.get(f"{API_URL}/account/{name_orig}/history", timeout=5)
            status_resp = requests.get(f"{API_URL}/account/{name_orig}/spike-status", timeout=5)
        except requests.RequestException as e:
            st.error(f"API call failed: {e}")
            resp = status_resp = None

        if resp is not None and resp.status_code == 404:
            st.warning("No scored transactions for this account yet.")
        elif resp is not None and resp.ok and status_resp is not None and status_resp.ok:
            history = pd.DataFrame(resp.json()) 
            status = status_resp.json()

            m1, m2, m3 = st.columns(3)
            m1.metric("Spike right now?", "YES" if status["is_spike"] else "no")
            m2.metric("Current rate", f"{status['current_rate']:.3f}")
            m3.metric("Z-score", f"{status['z_score']:.2f}")

            if not history.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history["step"], y=history["score"], mode="lines+markers",
                    line=dict(color="#6366f1", width=2),
                    marker=dict(
                        color=[DECISION_COLORS.get(d, "#888") for d in history["decision"]],
                        size=8,
                    ),
                    name="risk score",
                ))
                fig.add_hline(y=status["baseline_mean"], line_dash="dot", line_color="#9aa4b2",
                              annotation_text="this account's baseline")
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis_title="step", yaxis_title="risk score")
                st.plotly_chart(fig, use_container_width=True)
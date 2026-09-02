"""
Streamlit dashboard for the Fraud-Spike Sentinel.

Run (with the API already running separately):
    streamlit run app/dashboard.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Fraud-Spike Sentinel", layout="wide")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
DATA_PATH = st.sidebar.text_input("Replay data file", value="data/paysim.csv")

st.title("Fraud-Spike Sentinel")
st.caption("AI Risk Manager -- Live Transaction Feed & Account Forensics")

if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "cursor" not in st.session_state:
    st.session_state.cursor = 0

metrics_path = Path("model/metrics.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    st.subheader("Held-out test metrics")
    cols = st.columns(6)
    cols[0].metric("Precision", f"{metrics['precision']:.3f}")
    cols[1].metric("Recall", f"{metrics['recall']:.3f}")
    cols[2].metric("F1", f"{metrics['f1']:.3f}")
    cols[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    cols[4].metric("Threshold", f"{metrics['threshold']:.2f}")
    saved_vs_naive = metrics["cost_naive_half"] - metrics["cost_at_threshold"]
    cols[5].metric("Cost saved vs threshold=0.5", f"₹{saved_vs_naive:,.0f}")
    st.caption(
        f"Trained {metrics['trained_at']} on {metrics['n_train']:,} rows, "
        f"tested on {metrics['n_test']:,} held-out rows split by time, not randomly."
    )
else:
    st.warning("No model/metrics.json found yet -- run `python -m src.train` first.")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Live Threat Feed")
    
    col_input, col_btn = st.columns([2, 1])
    with col_input:
        stream_size = st.number_input("Transactions to simulate", min_value=1, max_value=500, value=25, label_visibility="collapsed")
    with col_btn:
        trigger = st.button("Start Live Stream", use_container_width=True)

    # This is the container that we will overwrite rapidly to create the "live" animation
    feed_placeholder = st.empty()

    if trigger:
        try:
            df = pd.read_csv(DATA_PATH)
            session = requests.Session()
            
            start = st.session_state.cursor
            end = min(start + stream_size, len(df))
            batch = df.iloc[start:end]
            st.session_state.cursor = end

            # THE ANIMATION LOOP
            for _, r in batch.iterrows():
                payload = {
                    "nameOrig": r["nameOrig"], "step": int(r["step"]), "amount": float(r["amount"]),
                    "type": r["type"], "oldbalanceOrg": float(r["oldbalanceOrg"]),
                    "newbalanceOrig": float(r["newbalanceOrig"]), "nameDest": r["nameDest"],
                    "oldbalanceDest": float(r["oldbalanceDest"]), "newbalanceDest": float(r["newbalanceDest"]),
                }
                
                resp = session.post(f"{API_URL}/score", json=payload, timeout=5)
                if resp.status_code == 200:
                    st.session_state.alerts.insert(0, {**payload, **resp.json()})
                
                # DRAW THE UI FRAME
                with feed_placeholder.container():
                    feed = pd.DataFrame(st.session_state.alerts[:50])
                    threats = feed[feed['decision'] != 'ALLOW']
                    
                    # KPIs
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Scored Txns", len(feed))
                    k2.metric("Threats Blocked", len(threats))
                    k3.metric("Value Protected", f"₹{threats['amount'].sum():,.0f}")

                    # Chart
                    fig = px.pie(feed, names='decision', hole=0.7, color='decision',
                                 color_discrete_map={'BLOCK': '#ff4b4b', 'REVIEW': '#ffa421', 'MONITOR': '#ffe83f', 'ALLOW': '#00c04b'})
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
                    
                    # 🔥 UNIQUE DYNAMIC KEY TO FIX CRASH
                    st.plotly_chart(fig, use_container_width=True, key=f"donut_live_{len(st.session_state.alerts)}")

                    # Table
                    def style_decision(val):
                        colors = {'BLOCK': '#ff4b4b', 'REVIEW': '#ffa421', 'MONITOR': '#ffe83f', 'ALLOW': '#00c04b'}
                        return f"color: {colors.get(val, 'white')}; font-weight: bold"

                    styled_feed = feed[["nameOrig", "amount", "type", "score", "decision"]].style.map(style_decision, subset=['decision'])
                    st.dataframe(
                        styled_feed,
                        column_config={
                            "score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0.0, max_value=1.0),
                            "amount": st.column_config.NumberColumn("Amount", format="₹%d")
                        },
                        use_container_width=True, height=250
                    )
                
                # Pause for visual heartbeat effect
                time.sleep(0.15)
                
        except Exception as e:
            st.error(f"Stream interrupted: {e}")

    # If stream isn't actively running, draw the static last state
    elif st.session_state.alerts:
        with feed_placeholder.container():
            feed = pd.DataFrame(st.session_state.alerts[:50])
            threats = feed[feed['decision'] != 'ALLOW']
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Scored Txns", len(feed))
            k2.metric("Threats Blocked", len(threats))
            k3.metric("Value Protected", f"₹{threats['amount'].sum():,.0f}")

            fig = px.pie(feed, names='decision', hole=0.7, color='decision',
                         color_discrete_map={'BLOCK': '#ff4b4b', 'REVIEW': '#ffa421', 'MONITOR': '#ffe83f', 'ALLOW': '#00c04b'})
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
            
            # 🔥 STATIC KEY FOR IDLE STATE
            st.plotly_chart(fig, use_container_width=True, key="donut_static")

            def style_decision(val):
                colors = {'BLOCK': '#ff4b4b', 'REVIEW': '#ffa421', 'MONITOR': '#ffe83f', 'ALLOW': '#00c04b'}
                return f"color: {colors.get(val, 'white')}; font-weight: bold"

            styled_feed = feed[["nameOrig", "amount", "type", "score", "decision"]].style.map(style_decision, subset=['decision'])
            st.dataframe(
                styled_feed,
                column_config={
                    "score": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0.0, max_value=1.0),
                    "amount": st.column_config.NumberColumn("Amount", format="₹%d")
                },
                use_container_width=True, height=250
            )

with right:
    st.subheader("Account Forensics")
    name_orig = st.text_input("Account ID (nameOrig)", value="")
    
    if st.button("Check Forensics") and name_orig:
        try:
            resp = requests.get(f"{API_URL}/account/{name_orig}/spike-status", timeout=5)
            if resp.status_code == 404:
                st.warning("No scored transactions for this account yet.")
            else:
                resp.raise_for_status()
                data = resp.json()
                
                st.metric("System Verdict", "ACCOUNT TAKEOVER" if data["is_spike"] else "NORMAL BEHAVIOR")
                
                c1, c2 = st.columns(2)
                c1.metric("Current Velocity", f"{data['current_rate']:.3f}", delta=f"{data['current_rate'] - data['baseline_mean']:.3f}", delta_color="inverse")
                c2.metric("Z-Score", f"{data['z_score']:.1f}", help=">3.0 indicates a severe spike")
                
        except requests.RequestException as e:
            st.error(f"API call failed: {e}")
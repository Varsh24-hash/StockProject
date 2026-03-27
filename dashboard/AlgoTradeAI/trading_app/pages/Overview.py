"""
🏠 Overview — AlgoTrade AI
Main landing page: market snapshot + strategy KPIs
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_price_data, get_features,
                   run_ml, run_rl, perf_metrics, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED, CARD, BORD)

st.set_page_config(page_title="Overview · AlgoTrade AI",
                   page_icon="📊", layout="wide")
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    cfg = sidebar_controls()

ticker  = cfg["ticker"]
capital = cfg["capital"]
txn     = cfg["txn"]
rf      = cfg["rf"]
model   = cfg["model"]
name    = TICKERS[ticker][0]

feat    = get_features(ticker)
ml_df, ml_trades = run_ml(ticker, model, capital, txn)
rl_df, rl_trades = run_rl(ticker, capital, txn)
ml_m = perf_metrics(ml_df["Portfolio"], rf)
rl_m = perf_metrics(rl_df["Portfolio"], rf)
bh_m = perf_metrics(ml_df["BuyHold"],   rf)
last  = feat.iloc[-1]
prev  = feat.iloc[-2]
chg   = (last["Close"] - prev["Close"]) / prev["Close"] * 100

# ── Header ────────────────────────────────────────────────────────────────────
page_header("Portfolio", "Overview",
            f"{name}  ·  {ticker}  ·  Capital ${capital:,.0f}")

# ── Top KPI strip ─────────────────────────────────────────────────────────────
active = ml_m if cfg["engine"] != "RL Agent" else rl_m
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col, (label, val) in zip([k1,k2,k3,k4,k5,k6], active.items()):
    col.metric(label, val)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
left, right = st.columns([2.4, 1], gap="large")

with left:
    section_label("Equity Curve Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml_df.index, y=ml_df["Portfolio"],
        line=dict(color=OR, width=2.2), name=f"ML ({model.replace('_',' ').title()})"))
    fig.add_trace(go.Scatter(x=rl_df.index, y=rl_df["Portfolio"],
        line=dict(color=GRN, width=2.2), name="RL Agent (PPO)"))
    fig.add_trace(go.Scatter(x=ml_df.index, y=ml_df["BuyHold"],
        line=dict(color=CREAM, width=1.2, dash="dot"), opacity=0.5, name="Buy & Hold"))
    fig.update_layout(**base_layout("", h=360))
    st.plotly_chart(fig, use_container_width=True)

    section_label("Drawdown")
    dd_ml = (ml_df["Portfolio"] - ml_df["Portfolio"].cummax()) / ml_df["Portfolio"].cummax() * 100
    dd_rl = (rl_df["Portfolio"] - rl_df["Portfolio"].cummax()) / rl_df["Portfolio"].cummax() * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ml_df.index, y=dd_ml, fill="tozeroy",
        fillcolor="rgba(212,98,26,0.08)", line=dict(color=OR, width=1.3), name="ML Drawdown"))
    fig2.add_trace(go.Scatter(x=rl_df.index, y=dd_rl, fill="tozeroy",
        fillcolor="rgba(82,183,136,0.07)", line=dict(color=GRN, width=1.3), name="RL Drawdown"))
    fig2.update_layout(**base_layout("", h=220))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    section_label("Live Quote")
    chg_col = GRN if chg >= 0 else RED
    glass_card(f"""
      <div style="font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;color:{chg_col};line-height:1">
        {last['Close']:,.2f}
      </div>
      <div style="font-size:0.78rem;color:{chg_col};margin:0.3rem 0 1rem">
        {'▲' if chg>=0 else '▼'} {abs(chg):.2f}%  today
      </div>
      {kv('Open',   f"{last['Open']:,.2f}")}
      {kv('High',   f"{last['High']:,.2f}", GRN)}
      {kv('Low',    f"{last['Low']:,.2f}",  RED)}
      {kv('Volume', f"{last['Volume']:,}")}
    """)

    section_label("Strategy Metrics")
    rows = ""
    for label, m_val, r_val, b_val in zip(
        ["Total Return","Ann. Return","Sharpe","Volatility","Max Drawdown","Win Rate"],
        ml_m.values(), rl_m.values(), bh_m.values()
    ):
        rows += f"""
        <div style="display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;
                    gap:0;padding:0.42rem 0;border-bottom:1px solid rgba(180,80,20,0.09);
                    font-size:0.75rem;align-items:center">
          <span style="color:{MUTE};font-family:'Outfit',sans-serif">{label}</span>
          <span style="color:{OR2};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{m_val}</span>
          <span style="color:{GRN};font-family:'Playfair Display',serif;font-weight:600;text-align:right">{r_val}</span>
          <span style="color:{CREAM};font-family:'Playfair Display',serif;font-weight:600;text-align:right;opacity:0.55">{b_val}</span>
        </div>"""
    glass_card(f"""
      <div style="display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;gap:0;
                  padding:0 0 0.5rem;border-bottom:1px solid rgba(180,80,20,0.18);
                  font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;
                  color:#5A4030;font-family:'Outfit',sans-serif;font-weight:600">
        <span>Metric</span>
        <span style="text-align:right">ML</span>
        <span style="text-align:right">RL</span>
        <span style="text-align:right">B&H</span>
      </div>
      {rows}
    """)

    section_label("Signal Indicators")
    rsi = last["RSI"]
    rsi_kind = "red" if rsi > 70 else ("green" if rsi < 30 else "orange")
    rsi_lbl  = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    macd_kind= "green" if last["MACD"] > last["MACD_Sig"] else "red"
    macd_lbl = "Bullish" if last["MACD"] > last["MACD_Sig"] else "Bearish"
    glass_card(f"""
      {kv('RSI (14)', f"{rsi:.1f} &nbsp; {pill(rsi_lbl, rsi_kind)}")}
      {kv('MACD',     f"{last['MACD']:.3f} &nbsp; {pill(macd_lbl, macd_kind)}")}
      {kv('BB Width', f"{(last['BB_Up']-last['BB_Lo'])/last['BB_Mid']*100:.1f}%")}
      {kv('Ann. Vol', f"{last['Vol_20']*100:.1f}%")}
      {kv('MA Cross', pill('Golden' if last['MA_50']>last['MA_200'] else 'Death', 'green' if last['MA_50']>last['MA_200'] else 'red'))}
    """, small=True)

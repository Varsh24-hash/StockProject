"""
🎲 Monte Carlo Simulation
10,000 future price paths, confidence bands, final value distribution
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, get_features, sidebar_controls,
                   OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Monte Carlo · AlgoTrade AI",
                   page_icon="🎲", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker    = cfg["ticker"]
mc_n      = cfg["mc_n"]
var_c     = cfg["var_c"]
feat      = get_features(ticker)
rets      = feat["Return_1d"].dropna()
last_p    = feat["Close"].iloc[-1]
mu, sigma = rets.mean(), rets.std()

@st.cache_data
def sim(last_price, mu, sigma, n_sims, days=252, seed=7):
    np.random.seed(seed)
    paths       = np.zeros((n_sims, days))
    paths[:, 0] = last_price
    for t in range(1, days):
        shocks     = np.random.normal(mu, sigma, n_sims)
        paths[:, t]= paths[:, t-1] * np.exp(shocks)
    return paths

paths  = sim(last_p, mu, sigma, mc_n)
p5     = np.percentile(paths[:,-1], 5)
p25    = np.percentile(paths[:,-1], 25)
p50    = np.percentile(paths[:,-1], 50)
p75    = np.percentile(paths[:,-1], 75)
p95    = np.percentile(paths[:,-1], 95)
var_mc = np.percentile((paths[:,-1]/last_p - 1)*100, (1-var_c)*100)

page_header("Monte Carlo", "Simulation",
            f"{mc_n:,} paths  ·  252-day horizon  ·  Geometric Brownian Motion")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Current Price",   f"{last_p:,.2f}")
c2.metric("5th Percentile",  f"{p5:,.2f}", f"{(p5/last_p-1)*100:+.1f}%")
c3.metric("Median Path",     f"{p50:,.2f}", f"{(p50/last_p-1)*100:+.1f}%")
c4.metric("95th Percentile", f"{p95:,.2f}", f"{(p95/last_p-1)*100:+.1f}%")
c5.metric("MC VaR (annual)", f"{var_mc:.2f}%", "252-day horizon")

st.markdown("<br>", unsafe_allow_html=True)

# ── Fan chart ─────────────────────────────────────────────────────────────────
section_label("Simulated Price Paths — Fan Chart")
days_ax = np.arange(paths.shape[1])
fig = go.Figure()

# Sample paths (faint)
idx = np.random.choice(mc_n, min(300, mc_n), replace=False)
for i in idx:
    fig.add_trace(go.Scatter(x=days_ax, y=paths[i],
        mode="lines", line=dict(color=OR, width=0.25), opacity=0.10,
        showlegend=False, hoverinfo="skip"))

# Percentile bands
for pct, col, name, w in [
    (5,  RED,   "5th pct",  2.0),
    (25, OR,    "25th pct", 1.4),
    (50, CREAM, "Median",   2.2),
    (75, OR,    "75th pct", 1.4),
    (95, GRN,   "95th pct", 2.0),
]:
    band = np.percentile(paths, pct, axis=0)
    fig.add_trace(go.Scatter(x=days_ax, y=band,
        line=dict(color=col, width=w), name=name))

fig.add_hline(y=last_p, line_dash="dash", line_color=GOLD, line_width=1.0,
              annotation_text="Entry price",
              annotation_font=dict(color=GOLD, size=10, family="JetBrains Mono"))
fig.update_layout(**base_layout("", h=420))
fig.update_xaxes(title_text="Trading days", title_font=dict(size=10))
fig.update_yaxes(title_text="Price", title_font=dict(size=10))
st.plotly_chart(fig, use_container_width=True)

# ── Final distribution ────────────────────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    section_label("Distribution of Final Values (Day 252)")
    final_rets = (paths[:,-1] / last_p - 1) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=paths[:,-1], nbinsx=100,
        marker_color=OR, opacity=0.70, name="Final price"))
    for val, col, lbl in [(p5,RED,"P5"),(p50,CREAM,"P50"),(p95,GRN,"P95")]:
        fig2.add_vline(x=val, line_dash="dash", line_color=col, line_width=1.2,
            annotation_text=lbl, annotation_font=dict(color=col, size=10, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=320))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    section_label("Scenario Summary")
    scenarios = [
        ("Bear (P5)",   p5,  (p5/last_p-1)*100,  "red"),
        ("Low (P25)",   p25, (p25/last_p-1)*100,  "orange"),
        ("Base (P50)",  p50, (p50/last_p-1)*100,  "gold"),
        ("High (P75)",  p75, (p75/last_p-1)*100,  "green"),
        ("Bull (P95)",  p95, (p95/last_p-1)*100,  "green"),
    ]
    rows_html = ""
    for name, price, ret, kind in scenarios:
        rows_html += f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 0.8fr;gap:0;
                    padding:0.5rem 0;border-bottom:1px solid rgba(180,80,20,0.1);
                    align-items:center">
          <span style="font-family:'Outfit',sans-serif;font-size:0.78rem;color:{MUTE}">{name}</span>
          <span style="font-family:'Playfair Display',serif;font-size:0.9rem;color:{CREAM};font-weight:600">{price:,.2f}</span>
          <span>{pill(f"{'+'if ret>=0 else ''}{ret:.1f}%", kind)}</span>
        </div>"""
    glass_card(rows_html)

    section_label("Distribution Stats")
    glass_card(f"""
      {kv('Mean final price',  f"{paths[:,-1].mean():,.2f}")}
      {kv('Std dev',           f"{paths[:,-1].std():,.2f}")}
      {kv('Skewness',          f"{pd.Series(paths[:,-1]).skew():.3f}")}
      {kv('Kurtosis',          f"{pd.Series(paths[:,-1]).kurtosis():.3f}")}
      {kv('Prob > entry',      f"{(paths[:,-1]>last_p).mean()*100:.1f}%", GRN)}
      {kv('Prob > +20%',       f"{(paths[:,-1]>last_p*1.2).mean()*100:.1f}%", GRN)}
      {kv('Prob < -20%',       f"{(paths[:,-1]<last_p*0.8).mean()*100:.1f}%", RED)}
    """)

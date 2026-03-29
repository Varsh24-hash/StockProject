"""
⚠️ Risk & VaR
Value at Risk from real daily returns, volatility forecasting
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

st.set_page_config(page_title="Risk & VaR · AlgoTrade AI",
                   page_icon="⚠️", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker = cfg["ticker"]
var_c  = cfg["var_c"]

# ── Real returns from actual OHLCV data ───────────────────────────────────────
feat = get_features(ticker)
rets = feat["Return_1d"].dropna()

hist_var  = np.percentile(rets, (1 - var_c) * 100)
param_var = rets.mean() - 1.645 * rets.std()
cvar      = rets[rets <= hist_var].mean()
ann_vol   = rets.std() * np.sqrt(252) * 100

page_header("Risk", "Management",
            f"Value at Risk  ·  {var_c*100:.0f}% Confidence  ·  Volatility Forecasting")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Historical VaR",  f"{hist_var*100:.3f}%",  "Daily 95% loss limit")
c2.metric("Parametric VaR",  f"{param_var*100:.3f}%", "Normal distribution")
c3.metric("CVaR / ES",       f"{cvar*100:.3f}%",      "Expected shortfall")
c4.metric("Ann. Volatility", f"{ann_vol:.2f}%",       "Annualised")

st.markdown("<br>", unsafe_allow_html=True)

top_l, top_r = st.columns(2, gap="large")

with top_l:
    section_label("Daily Return Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets*100, nbinsx=70,
        marker_color=OR, opacity=0.72, name="Daily returns"))
    fig.add_vline(x=hist_var*100, line_dash="dash", line_color=RED,
        annotation_text=f"VaR {hist_var*100:.2f}%",
        annotation_font=dict(color=RED, size=10, family="JetBrains Mono"),
        annotation_position="top left")
    fig.add_vline(x=rets.mean()*100, line_dash="dot", line_color=GRN,
        annotation_text=f"Mean {rets.mean()*100:.3f}%",
        annotation_font=dict(color=GRN, size=10, family="JetBrains Mono"))
    fig.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig, use_container_width=True)

with top_r:
    section_label("Rolling 60-Day VaR")
    roll_var = rets.rolling(60).apply(
        lambda x: np.percentile(x, (1-var_c)*100)) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_var.index, y=roll_var,
        fill="tozeroy", fillcolor="rgba(217,95,75,0.09)",
        line=dict(color=RED, width=1.5), name=f"Rolling {var_c*100:.0f}% VaR"))
    fig2.add_hline(y=hist_var*100, line_dash="dot", line_color=OR, line_width=0.8)
    fig2.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig2, use_container_width=True)

# ── Volatility: realised vs EWMA predicted ────────────────────────────────────
section_label("Volatility — Realised vs Predicted")
vol_real = feat["Vol_20"] * 100
# EWMA prediction (lambda=0.94, like RiskMetrics) — no random noise
vol_pred = rets.ewm(span=20).std() * np.sqrt(252) * 100

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=feat.index, y=vol_real,
    line=dict(color=CREAM, width=1.3), opacity=0.8, name="Realised vol (20d)"))
fig3.add_trace(go.Scatter(x=vol_pred.index, y=vol_pred,
    line=dict(color=OR2, width=1.5, dash="dot"), name="EWMA predicted vol"))
fig3.update_layout(**base_layout("", h=260))
st.plotly_chart(fig3, use_container_width=True)

# ── VaR method comparison ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_label("VaR Method Comparison")
v1, v2, v3 = st.columns(3, gap="large")

with v1:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Historical VaR</div>
      <div class="stat-number" style="color:{RED}">{hist_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Sorts past daily returns<br>
        Takes the {(1-var_c)*100:.0f}th percentile<br>
        No distribution assumption<br>
        Best for fat-tailed returns
      </div>
    """)

with v2:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Parametric VaR</div>
      <div class="stat-number" style="color:{OR}">{param_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Assumes normal distribution<br>
        Uses μ − 1.645σ formula<br>
        Fast and analytical<br>
        Underestimates tail risk
      </div>
    """)

with v3:
    np.random.seed(7)
    mc_r   = np.random.normal(rets.mean(), rets.std(), 50_000)
    mc_var = np.percentile(mc_r, (1-var_c)*100)
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Monte Carlo VaR</div>
      <div class="stat-numb"""
⚠️ Risk & VaR
All values driven by sidebar: ticker, var_c (confidence level)
Changing any sidebar value immediately recalculates all metrics and charts.
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

st.set_page_config(page_title="Risk & VaR · AlgoTrade AI",
                   page_icon="⚠️", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

# ── All values from sidebar ───────────────────────────────────────────────────
ticker = cfg["ticker"]
var_c  = cfg["var_c"]   # confidence level — changing slider recalculates all VaR

# ── Real returns from actual price data ───────────────────────────────────────
feat = get_features(ticker)
rets = feat["Return_1d"].dropna()

# ── All VaR metrics recompute when var_c or ticker changes ────────────────────
hist_var  = np.percentile(rets, (1 - var_c) * 100)
param_var = rets.mean() - 1.645 * rets.std()   # fixed z for 95%; parametric
cvar_val  = rets[rets <= hist_var].mean()
ann_vol   = rets.std() * np.sqrt(252) * 100

# Recompute parametric var with correct z for chosen confidence
from scipy import stats as scipy_stats
z_score   = scipy_stats.norm.ppf(1 - var_c)    # e.g. -1.645 at 95%
param_var_dynamic = rets.mean() + z_score * rets.std()

page_header("Risk", "Management",
            f"{TICKERS[ticker][0]}  ·  {var_c*100:.0f}% Confidence  ·  Volatility Forecasting")

# ── KPIs — all update with var_c ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Historical VaR",
          f"{hist_var*100:.3f}%",
          f"Daily {var_c*100:.0f}% loss limit")
c2.metric("Parametric VaR",
          f"{param_var_dynamic*100:.3f}%",
          "Normal distribution")
c3.metric("CVaR / ES",
          f"{cvar_val*100:.3f}%",
          "Expected shortfall")
c4.metric("Ann. Volatility",
          f"{ann_vol:.2f}%",
          "Annualised")

st.markdown("<br>", unsafe_allow_html=True)

top_l, top_r = st.columns(2, gap="large")

with top_l:
    section_label("Daily Return Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rets * 100, nbinsx=70,
        marker_color=OR, opacity=0.72, name="Daily returns"))
    # VaR line updates with var_c
    fig.add_vline(x=hist_var * 100, line_dash="dash", line_color=RED,
        annotation_text=f"VaR ({var_c*100:.0f}%) {hist_var*100:.2f}%",
        annotation_font=dict(color=RED, size=10, family="JetBrains Mono"),
        annotation_position="top left")
    fig.add_vline(x=rets.mean() * 100, line_dash="dot", line_color=GRN,
        annotation_text=f"Mean {rets.mean()*100:.3f}%",
        annotation_font=dict(color=GRN, size=10, family="JetBrains Mono"))
    fig.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig, use_container_width=True)

with top_r:
    section_label(f"Rolling 60-Day VaR ({var_c*100:.0f}% confidence)")
    # Rolling VaR updates when var_c changes
    roll_var = rets.rolling(60).apply(
        lambda x: np.percentile(x, (1 - var_c) * 100)
    ) * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=roll_var.index, y=roll_var,
        fill="tozeroy", fillcolor="rgba(217,95,75,0.09)",
        line=dict(color=RED, width=1.5),
        name=f"Rolling {var_c*100:.0f}% VaR"))
    fig2.add_hline(y=hist_var * 100, line_dash="dot",
                   line_color=OR, line_width=0.8,
                   annotation_text=f"Overall VaR {hist_var*100:.2f}%",
                   annotation_font=dict(color=OR, size=9, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=340))
    st.plotly_chart(fig2, use_container_width=True)

# ── Volatility: realised vs EWMA ─────────────────────────────────────────────
section_label("Volatility — Realised vs Predicted")
vol_real = feat["Vol_20"] * 100
vol_pred = rets.ewm(span=20).std() * np.sqrt(252) * 100

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=feat.index, y=vol_real,
    line=dict(color=CREAM, width=1.3), opacity=0.8, name="Realised vol (20d)"))
fig3.add_trace(go.Scatter(x=vol_pred.index, y=vol_pred,
    line=dict(color=OR2, width=1.5, dash="dot"), name="EWMA predicted vol"))
fig3.update_layout(**base_layout("", h=260))
st.plotly_chart(fig3, use_container_width=True)

# ── VaR method comparison — all three update with var_c ──────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_label(f"VaR Method Comparison  ·  {var_c*100:.0f}% confidence")
v1, v2, v3 = st.columns(3, gap="large")

# Monte Carlo VaR also uses var_c
np.random.seed(7)
mc_r   = np.random.normal(rets.mean(), rets.std(), 50_000)
mc_var = np.percentile(mc_r, (1 - var_c) * 100)

with v1:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Historical VaR</div>
      <div class="stat-number" style="color:{RED}">{hist_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Sorts past daily returns<br>
        Takes the {(1-var_c)*100:.0f}th percentile<br>
        No distribution assumption<br>
        Best for fat-tailed returns
      </div>
    """)

with v2:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Parametric VaR</div>
      <div class="stat-number" style="color:{OR}">{param_var_dynamic*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        Assumes normal distribution<br>
        Uses μ + z({var_c*100:.0f}%)·σ formula<br>
        Fast and analytical<br>
        Underestimates tail risk
      </div>
    """)

with v3:
    glass_card(f"""
      <div class="section-label" style="margin-bottom:0.8rem">Monte Carlo VaR</div>
      <div class="stat-number" style="color:{GOLD}">{mc_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        50,000 simulated returns<br>
        {(1-var_c)*100:.0f}th percentile of simulation<br>
        Captures non-linearity<br>
        Used by major banks
      </div>
    """)er" style="color:{GOLD}">{mc_var*100:.3f}%</div>
      <div style="margin-top:0.8rem;font-size:0.75rem;color:{MUTE};line-height:1.8">
        50,000 simulated returns<br>
        Captures non-linearity<br>
        Most computationally heavy<br>
        Used by major banks
      </div>
    """)
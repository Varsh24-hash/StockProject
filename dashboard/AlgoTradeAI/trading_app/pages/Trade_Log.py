"""
📋 Trade Log
Full execution log with real P&L from buy/sell round-trips
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (inject_css, page_header, section_label, glass_card, kv, pill,
                   base_layout, TICKERS, run_ml, run_rl, calc_pnl,
                   sidebar_controls, OR, OR2, GOLD, CREAM, MUTE, GRN, RED)

st.set_page_config(page_title="Trade Log · AlgoTrade AI",
                   page_icon="📋", layout="wide")
inject_css()

with st.sidebar:
    cfg = sidebar_controls()

ticker  = cfg["ticker"]
model   = cfg["model"]
capital = cfg["capital"]
txn     = cfg["txn"]

ml_df, ml_tr = run_ml(ticker, model, capital, txn)
rl_df, rl_tr = run_rl(ticker, capital, txn)

# ── Compute real P&L for both strategies ──────────────────────────────────────
ml_tr = calc_pnl(ml_tr, txn)
rl_tr = calc_pnl(rl_tr, txn)

page_header("Trade", "Execution Log",
            f"{TICKERS[ticker][0]}  ·  All strategies  ·  Full history")

engine = st.radio("View trades from", ["ML Strategy", "RL Agent"], horizontal=True)
trades = ml_tr.copy() if engine == "ML Strategy" else rl_tr.copy()
port_df= ml_df if engine == "ML Strategy" else rl_df

st.markdown("<br>", unsafe_allow_html=True)

if len(trades) == 0:
    st.info("No trades generated for this configuration.")
    st.stop()

trades["Date"] = pd.to_datetime(trades["Date"])

buy_t     = trades[trades["Type"] == "BUY"]
sell_t    = trades[trades["Type"] == "SELL"]
# Only count SELL trades for P&L (each SELL closes a round-trip)
pnl_sells = trades[trades["Type"] == "SELL"]["PnL"]
wins      = (pnl_sells > 0).sum()
total_pnl = pnl_sells.sum()
n_trades  = len(pnl_sells)   # count of completed round-trips

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Total Trades",   len(trades))
c2.metric("Buy Orders",     len(buy_t))
c3.metric("Sell Orders",    len(sell_t))
c4.metric("Winning Trades", f"{wins} ({wins/max(n_trades,1)*100:.0f}%)")
c5.metric("Total P&L",      f"${total_pnl:,.0f}")
c6.metric("Avg P&L/Trade",  f"${pnl_sells.mean() if n_trades else 0:,.0f}")

st.markdown("<br>", unsafe_allow_html=True)

top_l, top_r = st.columns([1.8, 1], gap="large")

with top_l:
    section_label("Cumulative P&L")
    cum_pnl = trades["PnL"].cumsum()
    bar_col = [GRN if v >= 0 else RED for v in trades["PnL"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trades["Date"], y=trades["PnL"],
        marker_color=bar_col, opacity=0.65, name="Trade P&L"))
    fig.add_trace(go.Scatter(x=trades["Date"], y=cum_pnl,
        line=dict(color=OR, width=2.2), name="Cumulative P&L", yaxis="y2"))
    fig.update_layout(**base_layout("", h=340))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right",
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(size=9, family="JetBrains Mono", color=OR)))
    st.plotly_chart(fig, use_container_width=True)

    section_label("P&L Distribution")
    pnl_nonzero = trades[trades["PnL"] != 0]["PnL"]
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=pnl_nonzero, nbinsx=30,
        marker_color=OR, opacity=0.7, name="P&L per trade"))
    fig2.add_vline(x=0, line_color=MUTE, line_width=1, line_dash="dot")
    if len(pnl_nonzero):
        fig2.add_vline(x=pnl_nonzero.mean(), line_color=GRN, line_width=1.2,
            line_dash="dash",
            annotation_text=f"Avg ${pnl_nonzero.mean():,.0f}",
            annotation_font=dict(color=GRN, size=9, family="JetBrains Mono"))
    fig2.update_layout(**base_layout("", h=240))
    st.plotly_chart(fig2, use_container_width=True)

with top_r:
    section_label("Trade Stats")
    win_pnl  = pnl_sells[pnl_sells > 0].mean() if (pnl_sells > 0).any() else 0
    loss_pnl = pnl_sells[pnl_sells < 0].mean() if (pnl_sells < 0).any() else 0
    pf = abs(win_pnl / loss_pnl) if loss_pnl != 0 else 0
    glass_card(f"""
      {kv('Total trades',     str(len(trades)))}
      {kv('Win rate',         f"{wins/max(n_trades,1)*100:.1f}%",   GRN)}
      {kv('Avg win',          f"${win_pnl:,.0f}",                   GRN)}
      {kv('Avg loss',         f"${loss_pnl:,.0f}",                  RED)}
      {kv('Profit factor',    f"{pf:.2f}",                          OR2)}
      {kv('Total P&L',        f"${total_pnl:,.0f}",
           GRN if total_pnl > 0 else RED)}
      {kv('Best trade',       f"${pnl_sells.max() if n_trades else 0:,.0f}",  GRN)}
      {kv('Worst trade',      f"${pnl_sells.min() if n_trades else 0:,.0f}",  RED)}
      {kv('Largest position', f"{trades['Shares'].max():,} shares")}
    """)

    section_label("Monthly Breakdown")
    trades["Month"] = trades["Date"].dt.to_period("M")
    monthly = trades.groupby("Month")["PnL"].sum().reset_index()
    monthly["Month"] = monthly["Month"].astype(str)
    cols = [GRN if v >= 0 else RED for v in monthly["PnL"]]
    fig3 = go.Figure(go.Bar(x=monthly["Month"], y=monthly["PnL"],
        marker_color=cols, opacity=0.75))
    fig3.update_layout(**base_layout("", h=260))
    fig3.update_xaxes(tickangle=45, tickfont=dict(size=8))
    st.plotly_chart(fig3, use_container_width=True)

# ── Full trade table ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_label("Full Trade History")
display = trades.copy()
display["Date"]  = display["Date"].dt.strftime("%Y-%m-%d")
display["Price"] = display["Price"].apply(lambda x: f"{x:,.2f}")
display["Value"] = (trades["Price"] * trades["Shares"]).apply(lambda x: f"${x:,.0f}")
display["PnL"]   = trades["PnL"].apply(lambda x: f"${x:+,.0f}" if x != 0 else "—")
display = display[["Date","Type","Price","Shares","Value","PnL"]]
st.dataframe(display, use_container_width=True, hide_index=True)
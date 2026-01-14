# ultimate_platform_ENHANCED.py
# AMAZING DASHBOARD (Drop-in) — Streamlit Cloud Safe
#
# Works with scanner_qullamaggie_enhanced_complete.py that exports:
#   SetupState, setup_store, run_scan(symbols, data_provider)
#
# Key features:
# - Beautiful summary + tabs
# - Full setup tables with TradingView links
# - "Trade forming" details (base levels, days in base, last update)
# - Interactive chart (Plotly candlestick via yfinance) for selected ticker
# - Positions panel (reads optional CSV export if present)
# - Telegram: /start menu + command replies (manual or auto-poll while page open)

from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
from datetime import datetime
import os

import streamlit as st
import pandas as pd

# Optional chart deps (already in your original app)
import plotly.graph_objects as go
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_module_attr(py_filename: str, module_name: str, attr_name: str):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        file_path = BASE_DIR / py_filename
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        return getattr(module, attr_name)


# --- Robust imports ---
run_scan = _load_module_attr("scanner_qullamaggie_enhanced_complete.py", "scanner_qullamaggie_enhanced_complete", "run_scan")
setup_store = _load_module_attr("scanner_qullamaggie_enhanced_complete.py", "scanner_qullamaggie_enhanced_complete", "setup_store")
SetupState = _load_module_attr("scanner_qullamaggie_enhanced_complete.py", "scanner_qullamaggie_enhanced_complete", "SetupState")

create_telegram_bot = _load_module_attr("telegram_bot_fixed.py", "telegram_bot_fixed", "create_telegram_bot")

# Tickers
def _fallback_tickers():
    return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]

try:
    from complete_tickers import get_all_tickers  # type: ignore
except Exception:
    get_all_tickers = _fallback_tickers


# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="Stocks Scanner", layout="wide")
st.title("🚀 Stocks Scanner — Qullamaggie System Pro")
st.caption("Monitoring → Entry → Invalidation with TradingView links, charts, positions, and Telegram commands.")


# ---------------------------
# Data provider
# ---------------------------
@st.cache_data(ttl=60)
def _fetch_ohlcv_yf(symbol: str, period: str = "6mo", interval: str = "1d") -> dict | None:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return {
            "close": df["Close"].astype(float).tolist(),
            "high": df["High"].astype(float).tolist(),
            "low": df["Low"].astype(float).tolist(),
            "volume": df["Volume"].astype(float).tolist() if "Volume" in df.columns else None,
        }
    except Exception:
        return None


def data_provider(symbol: str):
    # Use yfinance by default; replace with your live feed if you have one.
    return _fetch_ohlcv_yf(symbol)


# ---------------------------
# Helpers
# ---------------------------
def tv_link(symbol: str, exchange: str = "NASDAQ") -> str:
    sym = f"{exchange}:{symbol}" if ":" not in symbol else symbol
    return f"https://www.tradingview.com/chart/?symbol={sym}"


def to_df() -> pd.DataFrame:
    if not setup_store:
        return pd.DataFrame()
    df = pd.DataFrame(list(setup_store.values()))
    # friendly ordering
    preferred = [
        "symbol", "state", "last_price",
        "base_low", "base_high", "entry_price",
        "days_in_base", "invalid_reason", "last_state_change"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    if "last_state_change" in df.columns:
        df = df.sort_values("last_state_change", ascending=False)
    return df


def filter_state(df: pd.DataFrame, state) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["state"] == state].copy()


def counts(df: pd.DataFrame):
    if df.empty:
        return 0, 0, 0, 0, 0
    total = len(df)
    forming = int((df["state"] == SetupState.SETUP_FORMING).sum())
    monitoring = int((df["state"] == SetupState.MONITORING).sum())
    entries = int((df["state"] == SetupState.ENTRY_TRIGGERED).sum())
    invalid = int((df["state"] == SetupState.INVALIDATED).sum())
    return total, forming, monitoring, entries, invalid


def add_links(df: pd.DataFrame, exchange: str = "NASDAQ") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["tradingview"] = out["symbol"].apply(lambda s: tv_link(s, exchange))
    return out


def render_table(df: pd.DataFrame, empty_msg: str):
    if df.empty:
        st.info(empty_msg)
        return
    df2 = df.copy()
    if "tradingview" in df2.columns:
        # show as clickable markdown link
        df2["tradingview"] = df2["tradingview"].apply(lambda u: f"[Chart]({u})")
        st.dataframe(
            df2,
            use_container_width=True,
            column_config={"tradingview": st.column_config.MarkdownColumn("TradingView")},
        )
    else:
        st.dataframe(df2, use_container_width=True)


def candlestick(symbol: str, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        st.warning("No chart data.")
        return
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("⚙️ Scanner")

    universe = st.selectbox("Universe", ["Nasdaq + S&P500 list", "Small test list"], index=0)
    max_symbols = st.slider("Max symbols", 25, 2000, 400, 25)
    exchange = st.selectbox("TradingView exchange prefix", ["NASDAQ", "NYSE", "AMEX"], index=0)

    st.divider()
    st.header("🔔 Telegram")

    token = st.text_input("Bot token", type="password", value=os.getenv("TELEGRAM_TOKEN", ""))
    chat_id = st.text_input("Chat ID", value=os.getenv("TELEGRAM_CHAT_ID", ""))

    bot = None
    if token and chat_id:
        try:
            bot = create_telegram_bot(token, chat_id)
            st.success("Telegram ready ✅")
            if st.button("📌 Install /start menu"):
                bot.install_commands()
                st.success("Commands installed ✅")
        except Exception as e:
            st.error("Telegram init failed")
            st.exception(e)
            bot = None
    else:
        st.info("Enter token + chat id to enable Telegram.")

    auto_poll = st.checkbox("Auto-process Telegram commands (while page open)", value=False)
    poll_seconds = st.slider("Poll interval (sec)", 5, 60, 15, 5)

# Determine symbols
all_syms = get_all_tickers()
if universe == "Small test list":
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]
else:
    symbols = list(dict.fromkeys(all_syms))[:max_symbols]

# Auto refresh (optional)
if auto_poll and bot:
    # streamlit built-in rerun timer
    st.session_state["_last_autopoll"] = st.session_state.get("_last_autopoll", 0)
    now = datetime.utcnow().timestamp()
    if now - st.session_state["_last_autopoll"] >= poll_seconds:
        st.session_state["_last_autopoll"] = now
        # process commands (below)

# ---------------------------
# Run scan row
# ---------------------------
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
with col1:
    do_scan = st.button("🚀 Run Scan", use_container_width=True)
with col2:
    send_snapshot = st.button("📤 Send Snapshot to Telegram", use_container_width=True, disabled=(bot is None))
with col3:
    process_cmds_btn = st.button("💬 Process Telegram Commands", use_container_width=True, disabled=(bot is None))
with col4:
    st.caption(f"Loaded symbols: **{len(symbols)}** • Tracked: **{len(setup_store)}**")

if do_scan:
    with st.spinner("Scanning…"):
        run_scan(symbols, data_provider)
    st.success("Scan complete.")

df_all = add_links(to_df(), exchange=exchange)
total, forming, monitoring, entries, invalid = counts(df_all)

# ---------------------------
# Summary cards
# ---------------------------
st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tracked", total)
c2.metric("🔵 Forming", forming)
c3.metric("🟡 Monitoring", monitoring)
c4.metric("🟢 Entries", entries)
c5.metric("🔴 Invalid", invalid)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🏆 Best Setups", "🟡 Monitoring", "🔵 Forming", "🟢 Entries", "🔴 Invalidated", "📈 Charts / Positions"]
)

with tab1:
    st.subheader("🏆 Best Setups (all states)")
    if df_all.empty:
        st.info("Run a scan to populate setups.")
    else:
        # sort by state priority and recency
        state_order = {
            SetupState.ENTRY_TRIGGERED: 0,
            SetupState.MONITORING: 1,
            SetupState.SETUP_FORMING: 2,
            SetupState.INVALIDATED: 3,
            SetupState.NONE: 4,
        }
        df_rank = df_all.copy()
        df_rank["state_rank"] = df_rank["state"].map(state_order).fillna(9)
        if "last_state_change" in df_rank.columns:
            df_rank = df_rank.sort_values(["state_rank", "last_state_change"], ascending=[True, False])
        render_table(df_rank.drop(columns=["state_rank"], errors="ignore").head(200), "No setups found.")

with tab2:
    st.subheader("🟡 Monitoring — these are the ones to watch")
    df = filter_state(df_all, SetupState.MONITORING)
    render_table(df, "No monitoring setups right now.")

with tab3:
    st.subheader("🔵 Setup Forming — early candidates")
    df = filter_state(df_all, SetupState.SETUP_FORMING)
    render_table(df, "No setups forming right now.")

with tab4:
    st.subheader("🟢 Entry Triggered — ready to trade")
    df = filter_state(df_all, SetupState.ENTRY_TRIGGERED)
    render_table(df, "No entry triggers right now.")

with tab5:
    st.subheader("🔴 Invalidated — removed from watch")
    df = filter_state(df_all, SetupState.INVALIDATED)
    render_table(df, "No invalidations right now.")

with tab6:
    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("📈 Interactive Chart")
        if df_all.empty:
            st.info("Run a scan, then pick a ticker.")
        else:
            pick = st.selectbox("Pick a ticker", df_all["symbol"].tolist(), index=0)
            st.markdown(f"**TradingView:** [Open Chart]({tv_link(pick, exchange)})")
            candlestick(pick, period="6mo", interval="1d")

    with colB:
        st.subheader("📌 Setup Details")
        if df_all.empty:
            st.info("No setup selected.")
        else:
            row = df_all[df_all["symbol"] == pick].iloc[0].to_dict()
            st.write({
                "state": row.get("state"),
                "last_price": row.get("last_price"),
                "base_low": row.get("base_low"),
                "base_high": row.get("base_high"),
                "entry_price": row.get("entry_price"),
                "days_in_base": row.get("days_in_base"),
                "last_state_change": row.get("last_state_change"),
                "invalid_reason": row.get("invalid_reason"),
            })

        st.divider()
        st.subheader("💼 Positions (optional)")
        # If you export positions/tracker to CSV and keep it in repo root, it will show here
        csv_candidates = ["positions.csv", "2026-01-14T19-07_export.csv"]
        loaded = False
        for fn in csv_candidates:
            p = BASE_DIR / fn
            if p.exists():
                try:
                    pdf = pd.read_csv(p)
                    st.caption(f"Loaded: {fn}")
                    st.dataframe(pdf, use_container_width=True, height=260)
                    loaded = True
                    break
                except Exception:
                    pass
        if not loaded:
            st.info("No positions CSV found. (Optional) Add positions.csv to show positions here.")


# ---------------------------
# Telegram snapshot / commands
# ---------------------------
def _snapshot_text() -> str:
    return (
        "📊 Scanner Snapshot\n"
        f"Tracked: {total}\n"
        f"🔵 Forming: {forming}\n"
        f"🟡 Monitoring: {monitoring}\n"
        f"🟢 Entries: {entries}\n"
        f"🔴 Invalid: {invalid}\n"
        f"⏱ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

def _process_telegram():
    if not bot:
        return
    cmds = bot.read_commands()
    if not cmds:
        return
    # Provide context to command processor
    bot.reply_to_commands(
        commands=cmds,
        df=df_all,
        snapshot_text=_snapshot_text(),
        exchange=exchange,
    )

# send snapshot
if send_snapshot and bot:
    bot.send_text(_snapshot_text(), disable_web_page_preview=True)
    st.success("Sent snapshot to Telegram ✅")

# manual process
if process_cmds_btn and bot:
    _process_telegram()
    st.success("Processed Telegram commands ✅")

# auto poll if enabled (runs on page refresh)
if auto_poll and bot:
    _process_telegram()

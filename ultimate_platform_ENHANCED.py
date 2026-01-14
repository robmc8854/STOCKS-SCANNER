# ultimate_platform_ENHANCED.py
# FULL DROP-IN FILE — Streamlit Cloud Safe
# Fixes import issues for BOTH telegram_bot_fixed.py and scanner_qullamaggie_enhanced_complete.py
# Adds dashboard-like Setup Monitor sections + Telegram commands (processed on-demand)

from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
from datetime import datetime

import streamlit as st
import pandas as pd

# ==========================================================
# PATH FIX (Streamlit Cloud import reliability)
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_module_attr(py_filename: str, module_name: str, attr_name: str):
    """
    Robustly load an attribute from a local .py file in the same folder,
    even when Streamlit Cloud import paths are inconsistent.
    """
    # 1) Try normal import first
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        pass

    # 2) Fallback: load directly from disk
    file_path = BASE_DIR / py_filename
    if not file_path.exists():
        raise ImportError(f"{py_filename} not found next to {Path(__file__).name}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, attr_name):
        raise ImportError(f"{py_filename} loaded, but missing '{attr_name}'")
    return getattr(module, attr_name)


# ==========================================================
# TELEGRAM IMPORT (robust)
# ==========================================================
create_telegram_bot = _load_module_attr(
    py_filename="telegram_bot_fixed.py",
    module_name="telegram_bot_fixed",
    attr_name="create_telegram_bot",
)

# ==========================================================
# SCANNER IMPORT (robust)
# ==========================================================
run_scan = _load_module_attr(
    py_filename="scanner_qullamaggie_enhanced_complete.py",
    module_name="scanner_qullamaggie_enhanced_complete",
    attr_name="run_scan",
)
setup_store = _load_module_attr(
    py_filename="scanner_qullamaggie_enhanced_complete.py",
    module_name="scanner_qullamaggie_enhanced_complete",
    attr_name="setup_store",
)
SetupState = _load_module_attr(
    py_filename="scanner_qullamaggie_enhanced_complete.py",
    module_name="scanner_qullamaggie_enhanced_complete",
    attr_name="SetupState",
)

# ==========================================================
# TICKERS IMPORT (best effort)
# ==========================================================
def _get_all_tickers_fallback():
    return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]

try:
    from complete_tickers import get_all_tickers  # type: ignore
except Exception:
    get_all_tickers = _get_all_tickers_fallback


# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Stocks Scanner", layout="wide")
st.title("📊 STOCKS SCANNER — Qullamaggie System (Enhanced)")


# ==========================================================
# TELEGRAM CONFIG UI
# ==========================================================
with st.sidebar:
    st.header("🔔 Telegram")

    TELEGRAM_TOKEN = st.text_input("Bot Token", value="", type="password")
    TELEGRAM_CHAT_ID = st.text_input("Chat ID", value="")

    telegram_enabled = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)

    bot = None
    if telegram_enabled:
        try:
            bot = create_telegram_bot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            st.success("Telegram connected ✅")
        except Exception as e:
            st.error("Telegram connection failed ❌")
            st.exception(e)
            bot = None
    else:
        st.info("Enter token + chat id to enable Telegram.")


# ==========================================================
# DATA PROVIDER
# ==========================================================
def get_data_provider():
    """
    IMPORTANT:
    Replace _mock_data with your real market data fetcher if you have one.
    Must return dict with keys: close, high, low, (optional volume)
    """
    def _mock_data(symbol: str):
        import random
        close = [random.uniform(90, 120) for _ in range(120)]
        high = [c + random.uniform(0.5, 3.0) for c in close]
        low = [c - random.uniform(0.5, 3.0) for c in close]
        volume = [random.uniform(1e6, 8e6) for _ in range(120)]
        return {"close": close, "high": high, "low": low, "volume": volume}

    return _mock_data


data_provider = get_data_provider()


# ==========================================================
# CONTROLS
# ==========================================================
colA, colB, colC = st.columns([2, 1, 1])

with colA:
    universe = st.selectbox(
        "Universe",
        ["Nasdaq + S&P500 (from your tickers list)", "Small test list"],
        index=0,
    )

with colB:
    max_symbols = st.number_input("Max Symbols", min_value=10, max_value=5000, value=400)

with colC:
    send_telegram_snapshot = st.checkbox("Send Telegram Dashboard Snapshot after Scan", value=False)

symbols_all = get_all_tickers()
symbols_all = list(dict.fromkeys(symbols_all))  # de-dup

if universe == "Small test list":
    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]
else:
    symbols = symbols_all[: int(max_symbols)]

st.caption(f"Symbols loaded: **{len(symbols)}**")


# ==========================================================
# RUN SCAN
# ==========================================================
scan_btn = st.button("🚀 Run Scan", use_container_width=True)

if scan_btn:
    with st.spinner("Scanning…"):
        run_scan(symbols, data_provider)
    st.success(f"Scan complete — {len(setup_store)} symbols processed.")


# ==========================================================
# HELPERS
# ==========================================================
def _df_for_state(state) -> pd.DataFrame:
    rows = [s for s in setup_store.values() if s.get("state") == state]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    preferred = [
        "symbol",
        "state",
        "last_price",
        "base_low",
        "base_high",
        "entry_price",
        "days_in_base",
        "invalid_reason",
        "last_state_change",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    if "last_state_change" in df.columns:
        df = df.sort_values("last_state_change", ascending=False)
    return df


def _send_dashboard_snapshot_to_telegram():
    if not bot:
        return
    monitoring = sum(1 for s in setup_store.values() if s.get("state") == SetupState.MONITORING)
    forming = sum(1 for s in setup_store.values() if s.get("state") == SetupState.SETUP_FORMING)
    entries = sum(1 for s in setup_store.values() if s.get("state") == SetupState.ENTRY_TRIGGERED)
    invalid = sum(1 for s in setup_store.values() if s.get("state") == SetupState.INVALIDATED)

    msg = (
        "📊 Scanner Snapshot\n"
        f"🔵 Setup Forming: {forming}\n"
        f"🟡 Monitoring: {monitoring}\n"
        f"🟢 Entry Triggered: {entries}\n"
        f"🔴 Invalidated: {invalid}\n"
        f"⏱ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    bot.send_text(msg)


# ==========================================================
# DASHBOARD
# ==========================================================
st.divider()
st.header("🧠 Setup Monitor (State Engine)")

left, right = st.columns(2)

with left:
    st.subheader("🟡 Monitoring")
    df = _df_for_state(SetupState.MONITORING)
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("No monitoring setups right now.")

    st.subheader("🔵 Setup Forming")
    df = _df_for_state(SetupState.SETUP_FORMING)
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("No setups forming right now.")

with right:
    st.subheader("🟢 Entry Triggered")
    df = _df_for_state(SetupState.ENTRY_TRIGGERED)
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("No entry triggers right now.")

    st.subheader("🔴 Invalidated")
    df = _df_for_state(SetupState.INVALIDATED)
    st.dataframe(df, use_container_width=True) if not df.empty else st.info("No invalidations.")

st.divider()

# ==========================================================
# OPTIONAL TELEGRAM SNAPSHOT (after scan)
# ==========================================================
if send_telegram_snapshot and bot and scan_btn:
    try:
        _send_dashboard_snapshot_to_telegram()
        st.success("Telegram dashboard snapshot sent ✅")
    except Exception as e:
        st.error("Failed to send Telegram snapshot ❌")
        st.exception(e)

# ==========================================================
# TELEGRAM COMMANDS (ON DEMAND)
# ==========================================================
with st.sidebar:
    st.header("💬 Telegram Commands")
    st.caption("Send commands in Telegram, then click below to process once (Streamlit-safe).")
    process_cmds = st.button("📥 Process Telegram Commands", use_container_width=True)

if process_cmds:
    if not bot:
        st.warning("Telegram is not configured.")
    else:
        cmds = bot.read_commands()
        if not cmds:
            st.info("No new Telegram commands.")
        else:
            for cmd in cmds:
                cmd_clean = cmd.strip()

                if cmd_clean.startswith("/dashboard"):
                    _send_dashboard_snapshot_to_telegram()

                elif cmd_clean.startswith("/monitoring"):
                    bot.send_dataframe(_df_for_state(SetupState.MONITORING), title="🟡 Monitoring", max_rows=15)

                elif cmd_clean.startswith("/forming"):
                    bot.send_dataframe(_df_for_state(SetupState.SETUP_FORMING), title="🔵 Setup Forming", max_rows=15)

                elif cmd_clean.startswith("/entries"):
                    bot.send_dataframe(_df_for_state(SetupState.ENTRY_TRIGGERED), title="🟢 Entry Triggered", max_rows=15)

                elif cmd_clean.startswith("/invalidated"):
                    bot.send_dataframe(_df_for_state(SetupState.INVALIDATED), title="🔴 Invalidated", max_rows=15)

                elif cmd_clean.startswith("/ticker"):
                    parts = cmd_clean.split()
                    if len(parts) < 2:
                        bot.send_text("Usage: /ticker TSLA")
                    else:
                        sym = parts[1].upper()
                        setup = setup_store.get(sym)
                        if not setup:
                            bot.send_text(f"No data for {sym} (run scan first).")
                        else:
                            msg = (
                                f"📌 {sym}\n"
                                f"State: {setup.get('state')}\n"
                                f"Last Price: {setup.get('last_price')}\n"
                                f"Base: {setup.get('base_low')} – {setup.get('base_high')}\n"
                                f"Entry: {setup.get('entry_price')}\n"
                                f"Days in Base: {setup.get('days_in_base')}\n"
                                f"Reason: {setup.get('invalid_reason')}"
                            )
                            bot.send_text(msg, disable_web_page_preview=True)
                            bot.send_tradingview_chart(sym)

                else:
                    bot.send_text(
                        "Commands:\n"
                        "/dashboard\n"
                        "/monitoring\n"
                        "/forming\n"
                        "/entries\n"
                        "/invalidated\n"
                        "/ticker TSLA"
                    )

            st.success(f"Processed {len(cmds)} command(s).")

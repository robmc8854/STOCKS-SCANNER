"""
TELEGRAM BOT - DASHBOARD COMMANDS + ALERTS (UPGRADED)
- Keeps your existing "send Trade + TradingView link" flow
- Adds /commands that replicate most of the Streamlit dashboard
- Streamlit-friendly: can be polled opportunistically (no infinite loop required)
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Telegram limits: message max ~4096 chars
_TELEGRAM_MAX = 3900

DEFAULT_OFFSET_PATH = os.path.join("data", "telegram_offset.json")


def _now_hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _chunks(text: str, limit: int = _TELEGRAM_MAX) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        parts.append(text[:limit])
        text = text[limit:]
    return parts


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _fmt_money(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _tv_link(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    return f"https://www.tradingview.com/chart/?symbol={t}"


def _pre(text: str) -> str:
    # HTML pre formatting
    return f"<pre>{text}</pre>"


class TelegramDashboardBot:
    """
    Upgraded bot:
    - send_message / send_photo
    - send_trade_alert (compatible with your existing alert formatting)
    - process_updates(): handles /help /dashboard /monitoring /entries /invalidated /top /ticker /positions
    """

    def __init__(self, token: str, chat_id: str, offset_path: str = DEFAULT_OFFSET_PATH):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)
        self.offset_path = offset_path

        os.makedirs(os.path.dirname(self.offset_path), exist_ok=True)

        if self.enabled:
            ok = self.test_connection()
            self.enabled = ok

    # --------------------
    # Telegram primitives
    # --------------------
    def test_connection(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/getMe", timeout=8)
            if r.status_code != 200:
                logger.error("Telegram getMe failed: %s", r.text[:300])
                return False
            data = r.json()
            if not data.get("ok"):
                logger.error("Telegram getMe not ok: %s", data)
                return False
            logger.info("Telegram connected: @%s", data["result"].get("username"))
            return True
        except Exception as e:
            logger.error("Telegram connection error: %s", e)
            return False

    def send_message(self, message: str, parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
        if not self.enabled:
            return False
        ok_all = True
        for part in _chunks(message):
            payload = {
                "chat_id": self.chat_id,
                "text": part,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_preview,
            }
            try:
                r = requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=12)
                ok = (r.status_code == 200 and r.json().get("ok"))
                ok_all = ok_all and ok
                if not ok:
                    logger.error("Telegram sendMessage failed: %s", r.text[:300])
            except Exception as e:
                logger.error("Telegram send_message exception: %s", e)
                ok_all = False
        return ok_all

    def send_photo(self, photo_url: str, caption: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        try:
            payload = {"chat_id": self.chat_id, "photo": photo_url}
            if caption:
                payload["caption"] = caption
            r = requests.post(f"{self.base_url}/sendPhoto", data=payload, timeout=20)
            ok = (r.status_code == 200 and r.json().get("ok"))
            if not ok:
                logger.error("Telegram sendPhoto failed: %s", r.text[:300])
            return ok
        except Exception as e:
            logger.error("Telegram send_photo exception: %s", e)
            return False

    # --------------------
    # High level messages
    # --------------------
    def send_trade_alert(self, row: Dict[str, Any], validated: bool, reasons: str, shares: int = 0) -> bool:
        """
        row expected keys: ticker, setup_type, score, entry, stop, target_1R, target_2R
        """
        t = str(row.get("ticker", "")).upper()
        setup_type = row.get("setup_type", "SETUP")
        score = _safe_float(row.get("score", 0))
        entry = _safe_float(row.get("entry"))
        stop = _safe_float(row.get("stop"))
        t1 = _safe_float(row.get("target_1R"))
        t2 = _safe_float(row.get("target_2R"))

        risk = max(entry - stop, 0.00001)
        risk_pct = (risk / entry * 100) if entry else 0.0

        status_emoji = "✅" if validated else "❌"
        headline = "VALIDATED - READY TO TRADE" if validated else "REJECTED - DO NOT TRADE"

        msg = (
            f"🚀 <b>TRADE SIGNAL - {t}</b>\n\n"
            f"{status_emoji} <b>Status:</b> {headline}\n\n"
            f"📊 <b>Setup:</b> {setup_type}\n"
            f"⭐ <b>Score:</b> {score:.0f}/100\n\n"
            f"💰 <b>ENTRY:</b> ${entry:.2f}\n"
            f"🛑 <b>STOP:</b> ${stop:.2f} ({risk_pct:.1f}%)\n"
            f"🎯 <b>TARGET 1R:</b> ${t1:.2f}\n"
            f"🎯 <b>TARGET 2R:</b> ${t2:.2f}\n"
        )
        if shares:
            msg += f"\n📦 <b>Position Size:</b> {shares:,} shares"

        if reasons:
            # keep reasons concise
            msg += f"\n\n<b>🧾 Notes:</b>\n{reasons[:1200]}"

        ok = self.send_message(msg)
        # keep your known-good chart link behavior:
        self.send_message(f"📊 Chart: {_tv_link(t)}")
        return ok

    def send_state_events(self, events: List[Dict[str, Any]]) -> None:
        """Send concise state transitions (MONITORING/ENTRY/INVALIDATED)."""
        if not events:
            return
        lines = []
        for e in events[-20:]:
            t = e.get("ticker")
            st_from = e.get("prev_state")
            st_to = e.get("new_state")
            score = _safe_float(e.get("score", 0))
            cp = _safe_float(e.get("current_price", 0))
            entry = _safe_float(e.get("entry", 0))
            stop = _safe_float(e.get("stop", 0))
            if st_to == "ENTRY_TRIGGERED":
                emoji = "🟢"
            elif st_to == "INVALIDATED":
                emoji = "🔴"
            elif st_to == "MONITORING":
                emoji = "🟡"
            else:
                emoji = "ℹ️"
            lines.append(f"{emoji} {t} {st_from}→{st_to} | S:{score:.0f} CP:{cp:.2f} E:{entry:.2f} St:{stop:.2f}")
        msg = "📡 <b>SETUP STATE CHANGES</b>\n" + _pre("\n".join(lines))
        self.send_message(msg)

    def send_dashboard_snapshot(self, setups_df, scan_df=None) -> None:
        """
        setups_df: SetupTracker.to_dataframe() output
        scan_df: last scan dataframe (optional) for counts/scores
        """
        try:
            total_setups = 0 if setups_df is None else len(setups_df)
            monitoring = 0
            entries = 0
            invalid = 0
            if total_setups:
                monitoring = int((setups_df["state"] == "MONITORING").sum()) if "state" in setups_df.columns else 0
                entries = int((setups_df["state"] == "ENTRY_TRIGGERED").sum()) if "state" in setups_df.columns else 0
                invalid = int((setups_df["state"] == "INVALIDATED").sum()) if "state" in setups_df.columns else 0

            top_score = 0
            avg_score = 0.0
            if scan_df is not None and len(scan_df) > 0 and "score" in scan_df.columns:
                top_score = float(scan_df["score"].max())
                avg_score = float(scan_df["score"].mean())

            msg = (
                f"🧭 <b>DASHBOARD SNAPSHOT</b> ({_now_hhmm()})\n\n"
                f"🟡 Monitoring: <b>{monitoring}</b>\n"
                f"🟢 Entry Triggered: <b>{entries}</b>\n"
                f"🔴 Invalidated: <b>{invalid}</b>\n"
                f"📦 Total tracked: <b>{total_setups}</b>\n"
            )
            if scan_df is not None:
                msg += f"\n⭐ Top score: <b>{top_score:.0f}</b> | 📈 Avg: <b>{avg_score:.1f}</b>\n"

            msg += "\nCommands: /help  /monitoring  /entries  /top  /ticker TSLA"
            self.send_message(msg)
        except Exception as e:
            logger.error("send_dashboard_snapshot error: %s", e)

    # --------------------
    # Command processing
    # --------------------
    def _load_offset(self) -> int:
        try:
            if os.path.exists(self.offset_path):
                with open(self.offset_path, "r", encoding="utf-8") as f:
                    return int(json.load(f).get("offset", 0))
        except Exception:
            pass
        return 0

    def _save_offset(self, offset: int) -> None:
        try:
            os.makedirs(os.path.dirname(self.offset_path), exist_ok=True)
            with open(self.offset_path, "w", encoding="utf-8") as f:
                json.dump({"offset": int(offset)}, f)
        except Exception as e:
            logger.error("Could not save telegram offset: %s", e)

    def _get_updates(self, offset: int) -> Tuple[List[Dict[str, Any]], int]:
        try:
            params = {"timeout": 0, "offset": offset}
            r = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=12)
            if r.status_code != 200:
                return [], offset
            data = r.json()
            if not data.get("ok"):
                return [], offset
            updates = data.get("result", [])
            if not updates:
                return [], offset
            max_update_id = max(u.get("update_id", 0) for u in updates)
            return updates, max_update_id + 1
        except Exception as e:
            logger.error("getUpdates error: %s", e)
            return [], offset

    def process_updates(
        self,
        setup_tracker=None,
        last_scan_df=None,
        account: Any = None,
        max_updates: int = 10,
    ) -> None:
        """
        Call this from Streamlit (or your scanner) occasionally.
        It will read pending commands and respond.
        """
        if not self.enabled:
            return

        offset = self._load_offset()
        updates, new_offset = self._get_updates(offset)

        if not updates:
            return

        # keep only last N
        updates = updates[-max_updates:]

        setups_df = None
        if setup_tracker is not None:
            try:
                setups_df = setup_tracker.to_dataframe()
            except Exception:
                setups_df = None

        for u in updates:
            msg = u.get("message", {}) or {}
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue

            self._handle_command(text, setups_df, last_scan_df, account)

        # acknowledge processed
        self._save_offset(new_offset)

    def _handle_command(self, text: str, setups_df, scan_df, account: Any) -> None:
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/start", "/help"):
            self.send_message(
                "🤖 <b>Scanner Bot Commands</b>\n\n"
                "/dashboard - summary counts\n"
                "/monitoring - setups being watched\n"
                "/entries - entry triggered\n"
                "/invalidated - invalidated setups\n"
                "/top [n] - top setups by score\n"
                "/ticker SYMBOL - show one ticker details + chart\n"
                "/positions - open positions snapshot\n"
            )
            return

        if cmd in ("/dashboard", "/status"):
            self.send_dashboard_snapshot(setups_df, scan_df)
            return

        if cmd in ("/monitoring", "/entries", "/invalidated"):
            state = {
                "/monitoring": "MONITORING",
                "/entries": "ENTRY_TRIGGERED",
                "/invalidated": "INVALIDATED",
            }[cmd]
            self._send_state_table(setups_df, state)
            return

        if cmd == "/top":
            n = 10
            if len(parts) > 1:
                try:
                    n = max(1, min(25, int(parts[1])))
                except Exception:
                    n = 10
            self._send_top_table(setups_df, n=n)
            return

        if cmd == "/ticker":
            if len(parts) < 2:
                self.send_message("Usage: /ticker TSLA")
                return
            self._send_ticker(parts[1], setups_df)
            return

        if cmd == "/positions":
            if account is None:
                self.send_message("Positions not available in this context.")
                return
            self._send_positions(account)
            return

        self.send_message("Unknown command. Use /help")

    def _send_state_table(self, setups_df, state: str) -> None:
        if setups_df is None or len(setups_df) == 0:
            self.send_message("No tracked setups yet. Run a scan first.")
            return
        df = setups_df.copy()
        if "state" not in df.columns:
            self.send_message("Setup state data not available.")
            return
        df = df[df["state"] == state].sort_values(by=["score"], ascending=False)

        if len(df) == 0:
            self.send_message(f"No {state} setups.")
            return

        # keep concise
        df = df.head(20)
        lines = ["TICKER  SCORE  CP     ENTRY   STOP   TYPE"]
        for _, r in df.iterrows():
            lines.append(
                f"{str(r.get('ticker','')):<6} "
                f"{_safe_float(r.get('score',0)):>5.0f} "
                f"{_safe_float(r.get('current_price',0)):>6.2f} "
                f"{_safe_float(r.get('entry',0)):>6.2f} "
                f"{_safe_float(r.get('stop',0)):>6.2f} "
                f"{str(r.get('setup_type',''))[:10]}"
            )
        title = "🟡 MONITORING" if state == "MONITORING" else "🟢 ENTRY TRIGGERED" if state == "ENTRY_TRIGGERED" else "🔴 INVALIDATED"
        self.send_message(f"<b>{title}</b>\n" + _pre("\n".join(lines)))

    def _send_top_table(self, setups_df, n: int = 10) -> None:
        if setups_df is None or len(setups_df) == 0:
            self.send_message("No tracked setups yet. Run a scan first.")
            return
        df = setups_df.copy()
        if "score" not in df.columns:
            self.send_message("Score data not available.")
            return
        df = df.sort_values(by=["score"], ascending=False).head(n)

        lines = ["TICKER  STATE      SCORE  CP     ENTRY   STOP"]
        for _, r in df.iterrows():
            lines.append(
                f"{str(r.get('ticker','')):<6} "
                f"{str(r.get('state','')):<10} "
                f"{_safe_float(r.get('score',0)):>5.0f} "
                f"{_safe_float(r.get('current_price',0)):>6.2f} "
                f"{_safe_float(r.get('entry',0)):>6.2f} "
                f"{_safe_float(r.get('stop',0)):>6.2f}"
            )
        self.send_message("<b>🏆 TOP SETUPS</b>\n" + _pre("\n".join(lines)))

    def _send_ticker(self, ticker: str, setups_df) -> None:
        t = ticker.upper().strip()
        if setups_df is None or len(setups_df) == 0:
            self.send_message("No tracked setups yet. Run a scan first.")
            self.send_message(f"📊 Chart: {_tv_link(t)}")
            return
        df = setups_df.copy()
        df = df[df["ticker"] == t] if "ticker" in df.columns else df.iloc[0:0]
        if len(df) == 0:
            self.send_message(f"{t}: not currently tracked.\n📊 Chart: {_tv_link(t)}")
            return
        r = df.sort_values(by=["score"], ascending=False).iloc[0].to_dict()
        msg = (
            f"🔎 <b>{t}</b>\n"
            f"State: <b>{r.get('state')}</b>\n"
            f"Type: <b>{r.get('setup_type')}</b>\n"
            f"Score: <b>{_safe_float(r.get('score')):.0f}</b>\n"
            f"CP: <b>{_safe_float(r.get('current_price')):.2f}</b>\n"
            f"Entry: <b>{_safe_float(r.get('entry')):.2f}</b>\n"
            f"Stop: <b>{_safe_float(r.get('stop')):.2f}</b>\n"
        )
        if r.get("notes"):
            msg += f"\nNotes: {str(r.get('notes'))[:600]}"
        self.send_message(msg)
        self.send_message(f"📊 Chart: {_tv_link(t)}")

    def _send_positions(self, account: Any) -> None:
        try:
            account.update_positions()
            pos = getattr(account, "positions", []) or []
            if not pos:
                self.send_message("📊 <b>OPEN POSITIONS</b>\n\n✅ No open positions")
                return
            lines = ["TICKER  PNL£    PNL%   R   ENTRY  CUR   STOP"]
            for p in pos[:25]:
                lines.append(
                    f"{p.get('ticker',''):<6} "
                    f"{_safe_float(p.get('pnl',0)):>7.0f} "
                    f"{_safe_float(p.get('pnl_pct',0)):>6.2f} "
                    f"{_safe_float(p.get('r_multiple',0)):>4.2f} "
                    f"{_safe_float(p.get('entry_price',0)):>6.2f} "
                    f"{_safe_float(p.get('current_price',0)):>6.2f} "
                    f"{_safe_float(p.get('stop',0)):>6.2f}"
                )
            self.send_message("📊 <b>OPEN POSITIONS</b>\n" + _pre("\n".join(lines)))
        except Exception as e:
            self.send_message(f"Error getting positions: {e}")

# telegram_bot_fixed.py
# Upgraded Telegram Bot for STOCKS-SCANNER
#
# ✅ Keeps your current behavior:
# - send_trade_alert(...) then sends TradingView chart link
#
# ✅ Adds:
# - /start menu + /help
# - command processing that "replicates the dashboard" (tables, ticker detail, snapshot)
# - install_commands() to register commands in Telegram UI
#
# IMPORTANT:
# Streamlit Cloud can't run a bot 24/7 in the background.
# This bot is designed to be polled from Streamlit (manual button or auto-poll while page open).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import urllib.parse
import requests
import pandas as pd


@dataclass
class TelegramConfig:
    token: str
    chat_id: str


class TelegramBot:
    TELEGRAM_MAX_LEN = 3900

    def __init__(self, token: str, chat_id: str):
        self.cfg = TelegramConfig(token=token, chat_id=str(chat_id))
        self.base_url = f"https://api.telegram.org/bot{self.cfg.token}"
        self._last_update_id: Optional[int] = None

    # ------------------
    # HTTP helpers
    # ------------------
    def _get(self, method: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/{method}", params=params or {}, timeout=25)
            if not r.ok:
                return None
            data = r.json()
            if not data.get("ok"):
                return None
            return data
        except Exception:
            return None

    def _post(self, method: str, json_payload: Optional[Dict[str, Any]] = None, data_payload: Optional[Dict[str, Any]] = None) -> bool:
        try:
            url = f"{self.base_url}/{method}"
            if json_payload is not None:
                r = requests.post(url, json=json_payload, timeout=20)
            else:
                r = requests.post(url, data=data_payload or {}, timeout=20)
            return bool(r.ok)
        except Exception:
            return False

    def _chunk_text(self, text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        parts: List[str] = []
        buf = ""
        for line in text.splitlines():
            if len(buf) + len(line) + 1 > max_len:
                if buf:
                    parts.append(buf)
                    buf = ""
            if len(line) > max_len:
                for i in range(0, len(line), max_len):
                    parts.append(line[i:i+max_len])
            else:
                buf = (buf + "\n" + line) if buf else line
        if buf:
            parts.append(buf)
        return parts

    # ------------------
    # Messaging
    # ------------------
    def send_text(self, text: str, disable_web_page_preview: bool = False) -> bool:
        ok = True
        for part in self._chunk_text(text, self.TELEGRAM_MAX_LEN):
            payload = {
                "chat_id": self.cfg.chat_id,
                "text": part,
                "disable_web_page_preview": disable_web_page_preview,
            }
            ok = self._post("sendMessage", json_payload=payload) and ok
        return ok

    def send_dataframe(self, df: pd.DataFrame, title: str, max_rows: int = 15) -> None:
        if df is None or df.empty:
            self.send_text(f"{title}: none right now.", disable_web_page_preview=True)
            return
        d = df.copy().head(max_rows)
        if d.shape[1] > 9:
            d = d.iloc[:, :9]
        txt = d.to_string(index=False)
        self.send_text(f"{title}\n{txt}", disable_web_page_preview=True)

    # ------------------
    # TradingView
    # ------------------
    @staticmethod
    def tradingview_chart_link(symbol: str, exchange: str = "NASDAQ") -> str:
        sym = f"{exchange}:{symbol}" if ":" not in symbol else symbol
        return f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(sym)}"

    def send_tradingview_chart(self, symbol: str, exchange: str = "NASDAQ") -> bool:
        return self.send_text(f"📈 TradingView: {self.tradingview_chart_link(symbol, exchange)}", disable_web_page_preview=False)

    # ------------------
    # Keep your current trade alert behavior
    # ------------------
    def send_trade_alert(
        self,
        symbol: str,
        setup_type: str = "",
        score: Optional[float] = None,
        entry: Optional[float] = None,
        stop: Optional[float] = None,
        notes: str = "",
        exchange: str = "NASDAQ",
    ) -> None:
        parts = [f"🚨 Trade Found: {symbol}"]
        if setup_type:
            parts.append(f"Setup: {setup_type}")
        if score is not None:
            parts.append(f"Score: {score}")
        if entry is not None:
            parts.append(f"Entry: {entry}")
        if stop is not None:
            parts.append(f"Stop: {stop}")
        if notes:
            parts.append(notes)
        self.send_text("\n".join(parts), disable_web_page_preview=True)
        self.send_tradingview_chart(symbol, exchange=exchange)

    # ------------------
    # Commands UI (Telegram /start menu)
    # ------------------
    def install_commands(self) -> None:
        commands = [
            {"command": "start", "description": "Show menu"},
            {"command": "help", "description": "How to use"},
            {"command": "dashboard", "description": "Snapshot counts"},
            {"command": "monitoring", "description": "Monitoring setups"},
            {"command": "forming", "description": "Forming setups"},
            {"command": "entries", "description": "Entry triggered"},
            {"command": "invalidated", "description": "Invalidated"},
            {"command": "ticker", "description": "Ticker details: /ticker TSLA"},
            {"command": "top", "description": "Top setups: /top 20"},
        ]
        self._post("setMyCommands", json_payload={"commands": commands})

    def _menu_text(self) -> str:
        return (
            "👋 Welcome to Stocks Scanner Bot\n\n"
            "Commands:\n"
            "• /dashboard — snapshot counts\n"
            "• /monitoring — monitoring setups\n"
            "• /forming — setups forming\n"
            "• /entries — entry triggered\n"
            "• /invalidated — invalidated\n"
            "• /top 20 — top setups\n"
            "• /ticker TSLA — ticker detail + TradingView link\n\n"
            "Tip: run a scan in Streamlit so the bot has fresh data."
        )

    # ------------------
    # Polling (called from Streamlit)
    # ------------------
    def get_updates(self) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1
        data = self._get("getUpdates", params=params)
        if not data:
            return []
        results = data.get("result", [])
        if results:
            self._last_update_id = results[-1].get("update_id", self._last_update_id)
        return results

    def read_commands(self) -> List[str]:
        cmds: List[str] = []
        for upd in self.get_updates():
            msg = upd.get("message") or upd.get("edited_message") or {}
            text = (msg.get("text") or "").strip()
            if text.startswith("/"):
                cmds.append(text)
        return cmds

    # ------------------
    # Command router (called from Streamlit)
    # ------------------
    def reply_to_commands(self, commands: List[str], df: pd.DataFrame, snapshot_text: str, exchange: str = "NASDAQ") -> None:
        if df is None:
            df = pd.DataFrame()

        for cmd in commands:
            c = cmd.strip()
            c_low = c.lower()

            if c_low.startswith("/start"):
                self.send_text(self._menu_text(), disable_web_page_preview=True)

            elif c_low.startswith("/help"):
                self.send_text(self._menu_text(), disable_web_page_preview=True)

            elif c_low.startswith("/dashboard"):
                self.send_text(snapshot_text, disable_web_page_preview=True)

            elif c_low.startswith("/monitoring"):
                self.send_dataframe(df[df["state"].astype(str) == "MONITORING"], "🟡 Monitoring", max_rows=15)

            elif c_low.startswith("/forming"):
                self.send_dataframe(df[df["state"].astype(str) == "SETUP_FORMING"], "🔵 Setup Forming", max_rows=15)

            elif c_low.startswith("/entries"):
                self.send_dataframe(df[df["state"].astype(str) == "ENTRY_TRIGGERED"], "🟢 Entry Triggered", max_rows=15)

            elif c_low.startswith("/invalidated"):
                self.send_dataframe(df[df["state"].astype(str) == "INVALIDATED"], "🔴 Invalidated", max_rows=15)

            elif c_low.startswith("/top"):
                # /top 20
                n = 15
                parts = c.split()
                if len(parts) > 1:
                    try:
                        n = max(1, min(50, int(parts[1])))
                    except Exception:
                        n = 15
                self.send_dataframe(df.head(n), f"🏆 Top {n}", max_rows=n)

            elif c_low.startswith("/ticker"):
                parts = c.split()
                if len(parts) < 2:
                    self.send_text("Usage: /ticker TSLA", disable_web_page_preview=True)
                else:
                    sym = parts[1].upper()
                    sub = df[df["symbol"] == sym]
                    if sub.empty:
                        self.send_text(f"No data for {sym}. Run a scan first.", disable_web_page_preview=True)
                    else:
                        row = sub.iloc[0].to_dict()
                        msg = (
                            f"📌 {sym}\n"
                            f"State: {row.get('state')}\n"
                            f"Last: {row.get('last_price')}\n"
                            f"Base: {row.get('base_low')} – {row.get('base_high')}\n"
                            f"Entry: {row.get('entry_price')}\n"
                            f"Days in base: {row.get('days_in_base')}\n"
                            f"Reason: {row.get('invalid_reason')}"
                        )
                        self.send_text(msg, disable_web_page_preview=True)
                        self.send_tradingview_chart(sym, exchange=exchange)

            else:
                self.send_text("Unknown command. Send /start for menu.", disable_web_page_preview=True)


def create_telegram_bot(token: str, chat_id: str, **kwargs) -> TelegramBot:
    return TelegramBot(token=token, chat_id=str(chat_id))

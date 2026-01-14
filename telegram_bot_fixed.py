"""
TELEGRAM BOT - UPGRADED VERSION
- Keeps original alert sending methods (HTML support)
- Adds /start menu + command processing helpers
Designed to be polled from Streamlit (no always-on background required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    token: str
    chat_id: str


class TelegramBot:
    """
    Telegram bot for sending trading alerts + processing commands (polled).
    """
    TELEGRAM_MAX_LEN = 3900

    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        self.token = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and bool(token) and bool(chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id: Optional[int] = None

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _chunk(self, text: str) -> List[str]:
        if len(text) <= self.TELEGRAM_MAX_LEN:
            return [text]
        parts: List[str] = []
        buf = ""
        for line in text.splitlines():
            if len(buf) + len(line) + 1 > self.TELEGRAM_MAX_LEN:
                if buf:
                    parts.append(buf)
                    buf = ""
            if len(line) > self.TELEGRAM_MAX_LEN:
                for i in range(0, len(line), self.TELEGRAM_MAX_LEN):
                    parts.append(line[i:i+self.TELEGRAM_MAX_LEN])
            else:
                buf = (buf + "\n" + line) if buf else line
        if buf:
            parts.append(buf)
        return parts

    def _post(self, method: str, data: Dict[str, Any]) -> bool:
        if not self.enabled:
            logger.warning("⚠️ Telegram disabled - message not sent")
            return False
        try:
            url = f"{self.base_url}/{method}"
            r = requests.post(url, data=data, timeout=20)
            if not r.ok:
                logger.warning("Telegram API error: %s", r.text)
            return bool(r.ok)
        except Exception as e:
            logger.exception("Telegram post failed: %s", e)
            return False

    def _get(self, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            url = f"{self.base_url}/{method}"
            r = requests.get(url, params=params, timeout=25)
            if not r.ok:
                return None
            data = r.json()
            if not data.get("ok"):
                return None
            return data
        except Exception:
            return None

    # ----------------------------
    # Original send methods (compat)
    # ----------------------------
    def send_message(self, message: str, parse_mode: str = "HTML", disable_web_page_preview: bool = True) -> bool:
        """
        Send a message (supports HTML by default).
        Auto-chunks long messages.
        """
        ok = True
        for part in self._chunk(message):
            payload = {
                "chat_id": self.chat_id,
                "text": part,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview,
            }
            ok = self._post("sendMessage", payload) and ok
        return ok

    # Backward-compatible alias
    def send_telegram_alert(self, message: str, photo_url: Optional[str] = None) -> bool:
        ok = self.send_message(message, parse_mode="HTML", disable_web_page_preview=True)
        if photo_url:
            self._post("sendPhoto", {"chat_id": self.chat_id, "photo": photo_url, "caption": "Trade Setup Chart"})
        return ok

    def send_error(self, error: str) -> bool:
        return self.send_message(f"❌ <b>Error</b>\n\n<code>{error}</code>", parse_mode="HTML")

    def test_connection(self) -> bool:
        data = self._get("getMe", {})
        return bool(data)

    def send_scan_summary(self, count: int, market_condition: str = "") -> bool:
        msg = f"🔍 <b>SCAN COMPLETE</b>\n\n✅ Setups found: <b>{count}</b>"
        if market_condition:
            msg += f"\n📊 Market: <b>{market_condition}</b>"
        return self.send_message(msg)

    def send_setup_alert(self, ticker: str, setup_type: str, entry: float, stop: float, score: float, notes: str = "") -> bool:
        tv = f"https://www.tradingview.com/chart/?symbol=NASDAQ:{ticker}"
        msg = (
            f"🚨 <b>TRADE SETUP</b>\n\n"
            f"🎯 <b>{ticker}</b> | <b>{setup_type}</b>\n"
            f"⭐ Score: <b>{score:.0f}</b>\n"
            f"📍 Entry: <b>{entry:.2f}</b>\n"
            f"🛑 Stop: <b>{stop:.2f}</b>\n"
        )
        if notes:
            msg += f"📝 {notes}\n"
        msg += f"\n📈 <a href='{tv}'>TradingView Chart</a>"
        return self.send_message(msg, parse_mode="HTML", disable_web_page_preview=False)

    def send_position_alert(self, ticker: str, action: str, price: float, pnl: Optional[float] = None) -> bool:
        msg = f"💼 <b>POSITION UPDATE</b>\n\n{action}: <b>{ticker}</b> @ <b>{price:.2f}</b>"
        if pnl is not None:
            msg += f"\nP&L: <b>{pnl:.2f}</b>"
        return self.send_message(msg)

    # ----------------------------
    # New: /start menu + commands
    # ----------------------------
    def install_commands(self) -> bool:
        """
        Registers command list in Telegram UI (menu).
        """
        commands = [
            {"command": "start", "description": "Show menu"},
            {"command": "help", "description": "Help / commands"},
            {"command": "dashboard", "description": "Setup tracker snapshot"},
            {"command": "monitoring", "description": "Monitoring setups"},
            {"command": "forming", "description": "Forming setups"},
            {"command": "entries", "description": "Entry triggered"},
            {"command": "invalidated", "description": "Invalidated"},
            {"command": "ticker", "description": "Ticker detail: /ticker TSLA"},
        ]
        return self._post("setMyCommands", {"commands": str(commands).replace("'", '"')})  # JSON-ish

    def _menu_text(self) -> str:
        return (
            "👋 <b>Stocks Scanner Bot</b>\n\n"
            "Commands:\n"
            "• /dashboard — snapshot counts\n"
            "• /monitoring — monitoring setups\n"
            "• /forming — setups forming\n"
            "• /entries — entry triggered\n"
            "• /invalidated — invalidated\n"
            "• /ticker TSLA — ticker detail + TradingView link\n\n"
            "Tip: Run a scan in Streamlit so results are fresh."
        )

    def get_updates(self) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1
        data = self._get("getUpdates", params)
        if not data:
            return []
        res = data.get("result", [])
        if res:
            self._last_update_id = res[-1].get("update_id", self._last_update_id)
        return res

    def read_commands(self) -> List[str]:
        cmds: List[str] = []
        for upd in self.get_updates():
            msg = upd.get("message") or upd.get("edited_message") or {}
            text = (msg.get("text") or "").strip()
            if text.startswith("/"):
                cmds.append(text)
        return cmds

    def reply_to_commands(self, commands: List[str], df: pd.DataFrame, snapshot_text: Optional[str] = None) -> None:
        """
        Reply to a list of commands using provided tracker dataframe (ticker/state/etc).
        """
        if df is None:
            df = pd.DataFrame()

        def send_df(title: str, sub: pd.DataFrame, max_rows: int = 15):
            if sub is None or sub.empty:
                self.send_message(f"{title}: none right now.", parse_mode="HTML")
                return
            d = sub.head(max_rows).copy()
            # keep important cols
            cols = [c for c in ["ticker","state","setup_type","score","last_price","entry","stop","% to entry"] if c in d.columns]
            if cols:
                d = d[cols]
            text = d.to_string(index=False)
            self.send_message(f"<b>{title}</b>\n<code>{text}</code>", parse_mode="HTML")

        for cmd in commands:
            c = cmd.strip()
            low = c.lower()

            if low.startswith("/start") or low.startswith("/help"):
                self.send_message(self._menu_text(), parse_mode="HTML", disable_web_page_preview=True)

            elif low.startswith("/dashboard"):
                if snapshot_text:
                    self.send_message(snapshot_text, parse_mode="HTML")
                else:
                    # derive counts
                    if df.empty or "state" not in df.columns:
                        self.send_message("📊 <b>Dashboard</b>\n\nNo data yet. Run a scan.", parse_mode="HTML")
                    else:
                        counts = df["state"].value_counts().to_dict()
                        msg = "📊 <b>Dashboard</b>\n\n"
                        for k,v in counts.items():
                            msg += f"• {k}: <b>{v}</b>\n"
                        self.send_message(msg, parse_mode="HTML")

            elif low.startswith("/monitoring"):
                send_df("🟡 Monitoring", df[df.get("state","")=="MONITORING"])

            elif low.startswith("/forming"):
                send_df("🔵 Forming", df[df.get("state","")=="FORMING"])

            elif low.startswith("/entries"):
                send_df("🟢 Entries", df[df.get("state","")=="ENTRY_TRIGGERED"])

            elif low.startswith("/invalidated"):
                send_df("🔴 Invalidated", df[df.get("state","")=="INVALIDATED"])

            elif low.startswith("/ticker"):
                parts = c.split()
                if len(parts) < 2:
                    self.send_message("Usage: /ticker TSLA", parse_mode="HTML")
                else:
                    sym = parts[1].upper()
                    sub = df[df.get("ticker","")==sym] if not df.empty else pd.DataFrame()
                    if sub.empty:
                        self.send_message(f"No data for {sym}. Run a scan.", parse_mode="HTML")
                    else:
                        row = sub.iloc[0].to_dict()
                        tv = f"https://www.tradingview.com/chart/?symbol=NASDAQ:{sym}"
                        msg = (
                            f"📌 <b>{sym}</b>\n"
                            f"State: <b>{row.get('state')}</b>\n"
                            f"Setup: <b>{row.get('setup_type')}</b>\n"
                            f"Score: <b>{row.get('score')}</b>\n"
                            f"Last: <b>{row.get('last_price')}</b>\n"
                            f"Entry: <b>{row.get('entry')}</b>\n"
                            f"Stop: <b>{row.get('stop')}</b>\n"
                            f"\n📈 <a href='{tv}'>TradingView Chart</a>"
                        )
                        self.send_message(msg, parse_mode="HTML", disable_web_page_preview=False)
            else:
                self.send_message("Unknown command. Send /start for menu.", parse_mode="HTML")


def create_telegram_bot(token: str, chat_id: str, **kwargs) -> TelegramBot:
    return TelegramBot(token=token, chat_id=str(chat_id))

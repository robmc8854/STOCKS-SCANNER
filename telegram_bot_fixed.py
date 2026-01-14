# telegram_bot_fixed.py
# FULL DROP-IN FILE — Upgraded Telegram for STOCKS-SCANNER
#
# Goals:
# - Keep your existing behavior: trade alerts + TradingView chart link
# - Add dashboard-like commands support (/dashboard, /monitoring, /entries, etc.)
# - Be Streamlit-safe: no background threads required; "poll once" design
# - Be robust against Telegram message length limits (auto-chunking)
#
# Provides:
#   create_telegram_bot(token, chat_id)  ✅ required by your Streamlit app
#   TelegramBot.send_trade_alert(...)    ✅ keep existing workflow
#   TelegramBot.read_commands()          ✅ optional command polling
#   TelegramBot.send_dataframe(...)      ✅ replicate dashboard tables

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
    """
    Backward-compatible Telegram client.
    Works in Streamlit by calling methods on-demand (no loops needed).
    """

    TELEGRAM_MAX_LEN = 3900  # keep under hard limits

    def __init__(self, token: str, chat_id: str):
        self.cfg = TelegramConfig(token=token, chat_id=str(chat_id))
        self.base_url = f"https://api.telegram.org/bot{self.cfg.token}"
        self._last_update_id: Optional[int] = None  # used for polling commands

    # ------------------
    # Internal helpers
    # ------------------
    def _post(self, method: str, json_payload: Optional[Dict[str, Any]] = None, data_payload: Optional[Dict[str, Any]] = None) -> bool:
        try:
            url = f"{self.base_url}/{method}"
            if json_payload is not None:
                r = requests.post(url, json=json_payload, timeout=20)
            else:
                r = requests.post(url, data=data_payload, timeout=20)
            return bool(r.ok)
        except Exception:
            return False

    def _chunk_text(self, text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        chunks: List[str] = []
        lines = text.splitlines(keepends=False)
        buf = ""
        for line in lines:
            if len(buf) + len(line) + 1 > max_len:
                if buf:
                    chunks.append(buf)
                    buf = ""
            if len(line) > max_len:
                # hard split long line
                for i in range(0, len(line), max_len):
                    part = line[i:i+max_len]
                    if part:
                        if buf:
                            chunks.append(buf)
                            buf = ""
                        chunks.append(part)
            else:
                buf = (buf + "\n" + line) if buf else line
        if buf:
            chunks.append(buf)
        return chunks

    # ------------------
    # Low-level senders
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

    def send_photo_url(self, photo_url: str, caption: Optional[str] = None) -> bool:
        payload = {"chat_id": self.cfg.chat_id, "photo": photo_url}
        if caption:
            payload["caption"] = caption[:900]
        return self._post("sendPhoto", data_payload=payload)

    # ------------------
    # TradingView helper
    # ------------------
    @staticmethod
    def tradingview_chart_link(symbol: str, exchange: str = "NASDAQ") -> str:
        sym = f"{exchange}:{symbol}" if ":" not in symbol else symbol
        return f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(sym)}"

    def send_tradingview_chart(self, symbol: str, exchange: str = "NASDAQ") -> bool:
        link = self.tradingview_chart_link(symbol, exchange=exchange)
        return self.send_text(f"📈 TradingView: {link}", disable_web_page_preview=False)

    # ------------------
    # Backward-compatible "trade found" message
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

        # Keep your current workflow: alert + then TradingView link
        self.send_text("\n".join(parts), disable_web_page_preview=True)
        self.send_tradingview_chart(symbol, exchange=exchange)

    # ------------------
    # Dashboard-like tables
    # ------------------
    def send_dataframe(self, df: pd.DataFrame, title: str = "", max_rows: int = 15) -> None:
        if df is None or df.empty:
            self.send_text(f"{title}: none right now." if title else "None right now.")
            return

        # limit size
        df2 = df.head(max_rows).copy()

        # Compact columns: keep first ~9
        if df2.shape[1] > 9:
            df2 = df2.iloc[:, :9]

        txt = df2.to_string(index=False)
        msg = f"{title}\n{txt}" if title else txt
        self.send_text(msg, disable_web_page_preview=True)

    # ------------------
    # Commands (poll once)
    # ------------------
    def get_updates(self, timeout: int = 0) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/getUpdates"
            params: Dict[str, Any] = {"timeout": timeout}
            if self._last_update_id is not None:
                params["offset"] = self._last_update_id + 1

            r = requests.get(url, params=params, timeout=25)
            if not r.ok:
                return []
            data = r.json()
            if not data.get("ok"):
                return []
            results = data.get("result", [])
            if results:
                self._last_update_id = results[-1].get("update_id", self._last_update_id)
            return results
        except Exception:
            return []

    @staticmethod
    def _extract_text(update: Dict[str, Any]) -> Optional[str]:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return None
        return msg.get("text")

    def read_commands(self) -> List[str]:
        """
        Returns messages starting with '/' since the last poll.
        """
        commands: List[str] = []
        for upd in self.get_updates(timeout=0):
            text = self._extract_text(upd)
            if text and text.strip().startswith("/"):
                commands.append(text.strip())
        return commands


def create_telegram_bot(token: str, chat_id: str, **kwargs) -> TelegramBot:
    """
    Backward-compatible factory expected by ultimate_platform_ENHANCED.py
    """
    return TelegramBot(token=token, chat_id=str(chat_id))

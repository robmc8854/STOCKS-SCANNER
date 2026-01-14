# telegram_bot_fixed.py
# FULL DROP-IN FILE — Upgraded Telegram for STOCKS-SCANNER
#
# - Provides create_telegram_bot(token, chat_id) expected by Streamlit app
# - Keeps trade alert + TradingView chart link workflow
# - Adds send_dataframe for dashboard-style tables
# - Adds read_commands() for on-demand command processing (Streamlit-safe)
# - Auto-chunks long messages to avoid Telegram limits

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
    TELEGRAM_MAX_LEN = 3900  # safety margin under Telegram hard limit

    def __init__(self, token: str, chat_id: str):
        self.cfg = TelegramConfig(token=token, chat_id=str(chat_id))
        self.base_url = f"https://api.telegram.org/bot{self.cfg.token}"
        self._last_update_id: Optional[int] = None

    # ------------------
    # Utils
    # ------------------
    def _chunk_text(self, text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        chunks: List[str] = []
        lines = text.splitlines()
        buf = ""
        for line in lines:
            if len(buf) + len(line) + 1 > max_len:
                if buf:
                    chunks.append(buf)
                    buf = ""
            # split very long lines
            while len(line) > max_len:
                chunks.append(line[:max_len])
                line = line[max_len:]
            buf = (buf + "\n" + line) if buf else line
        if buf:
            chunks.append(buf)
        return chunks

    def _post_json(self, method: str, payload: Dict[str, Any]) -> bool:
        try:
            r = requests.post(f"{self.base_url}/{method}", json=payload, timeout=20)
            return bool(r.ok)
        except Exception:
            return False

    def _post_data(self, method: str, payload: Dict[str, Any]) -> bool:
        try:
            r = requests.post(f"{self.base_url}/{method}", data=payload, timeout=20)
            return bool(r.ok)
        except Exception:
            return False

    # ------------------
    # Senders
    # ------------------
    def send_text(self, text: str, disable_web_page_preview: bool = False) -> bool:
        ok = True
        for part in self._chunk_text(text, self.TELEGRAM_MAX_LEN):
            payload = {
                "chat_id": self.cfg.chat_id,
                "text": part,
                "disable_web_page_preview": disable_web_page_preview,
            }
            ok = self._post_json("sendMessage", payload) and ok
        return ok

    def send_photo_url(self, photo_url: str, caption: Optional[str] = None) -> bool:
        payload = {"chat_id": self.cfg.chat_id, "photo": photo_url}
        if caption:
            payload["caption"] = caption[:900]
        return self._post_data("sendPhoto", payload)

    # ------------------
    # TradingView
    # ------------------
    @staticmethod
    def tradingview_chart_link(symbol: str, exchange: str = "NASDAQ") -> str:
        sym = f"{exchange}:{symbol}" if ":" not in symbol else symbol
        return f"https://www.tradingview.com/chart/?symbol={urllib.parse.quote(sym)}"

    def send_tradingview_chart(self, symbol: str, exchange: str = "NASDAQ") -> bool:
        link = self.tradingview_chart_link(symbol, exchange=exchange)
        return self.send_text(f"📈 TradingView: {link}", disable_web_page_preview=False)

    # ------------------
    # Trade alerts (keep existing workflow)
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
    # Dashboard-style tables
    # ------------------
    def send_dataframe(self, df: pd.DataFrame, title: str = "", max_rows: int = 15) -> None:
        if df is None or df.empty:
            self.send_text(f"{title}: none right now." if title else "None right now.")
            return

        df2 = df.head(max_rows).copy()
        if df2.shape[1] > 10:
            df2 = df2.iloc[:, :10]

        text = df2.to_string(index=False)
        msg = f"{title}\n{text}" if title else text
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
        cmds: List[str] = []
        for upd in self.get_updates(timeout=0):
            text = self._extract_text(upd)
            if text and text.strip().startswith("/"):
                cmds.append(text.strip())
        return cmds


def create_telegram_bot(token: str, chat_id: str, **kwargs) -> TelegramBot:
    return TelegramBot(token=token, chat_id=str(chat_id))

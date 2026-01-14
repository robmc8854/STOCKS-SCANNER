# scanner_qullamaggie_enhanced_complete.py
"""
WORLD-CLASS (STABLE) BACKWARD-COMPAT SCANNER MODULE

This file is designed to stop Streamlit crashes and preserve your dashboard interface.

Exports expected by ultimate_platform_ENHANCED.py:
- UltraScannerEngine            (dashboard calls: scanner.run_full_scan() with NO args)
- SetupState, SetupTracker      (used by setup monitoring enhancements)
- tradingview_link() helper

Design goals:
- Never crash if dashboard forgets to pass a provider or symbols.
- Sensible defaults: tickers from complete_tickers.get_all_tickers() and data via yfinance.
- Returns a pandas DataFrame for scan results (dashboard-friendly).
- Keeps a persistent setup tracker for Monitoring -> Entry -> Invalidated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Tuple

import pandas as pd
import yfinance as yf


# -----------------------------
# TradingView helper
# -----------------------------
def tradingview_link(symbol: str, exchange: str = "NASDAQ") -> str:
    sym = f"{exchange}:{symbol}" if ":" not in symbol else symbol
    return f"https://www.tradingview.com/chart/?symbol={sym}"


# -----------------------------
# Default market data provider
# -----------------------------
def yf_provider(symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[Dict[str, List[float]]]:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        out = {
            "close": df["Close"].astype(float).tolist(),
            "high": df["High"].astype(float).tolist(),
            "low": df["Low"].astype(float).tolist(),
        }
        if "Volume" in df.columns:
            out["volume"] = df["Volume"].astype(float).tolist()
        return out
    except Exception:
        return None


# -----------------------------
# Setup state engine
# -----------------------------
class SetupState(str, Enum):
    NONE = "NONE"
    SETUP_FORMING = "SETUP_FORMING"
    MONITORING = "MONITORING"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    INVALIDATED = "INVALIDATED"


@dataclass
class SetupRecord:
    symbol: str
    state: str = SetupState.NONE.value
    setup_type: str = "Qullamaggie"
    score: float = 0.0
    entry: Optional[float] = None
    stop: Optional[float] = None
    current_price: Optional[float] = None
    base_high: Optional[float] = None
    base_low: Optional[float] = None
    days_in_base: int = 0
    invalid_reason: Optional[str] = None
    last_state_change: str = ""
    updated_at: str = ""
    miss_count: int = 0


class SetupTracker:
    """
    Persistent tracker driven by scan results (DataFrame).
    Call:
      - ingest_scan_results(df)
      - update_prices_and_states(optional_provider)
    """

    def __init__(self):
        self.records: Dict[str, SetupRecord] = {}

    def ingest_scan_results(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        now = datetime.utcnow().isoformat()

        for _, row in df.iterrows():
            sym = str(row.get("symbol") or row.get("ticker") or "").upper()
            if not sym:
                continue

            rec = self.records.get(sym) or SetupRecord(symbol=sym)
            rec.updated_at = now
            rec.miss_count = 0

            # Keep scan fields if present
            rec.setup_type = str(row.get("setup_type") or row.get("setup") or rec.setup_type)
            rec.score = float(row.get("score") or rec.score or 0.0)

            entry = row.get("entry", None)
            stop = row.get("stop", None)
            cp = row.get("current_price", None)

            rec.entry = float(entry) if entry not in (None, "", "nan") else rec.entry
            rec.stop = float(stop) if stop not in (None, "", "nan") else rec.stop
            rec.current_price = float(cp) if cp not in (None, "", "nan") else rec.current_price

            base_high = row.get("base_high", None)
            base_low = row.get("base_low", None)
            rec.base_high = float(base_high) if base_high not in (None, "", "nan") else rec.base_high
            rec.base_low = float(base_low) if base_low not in (None, "", "nan") else rec.base_low

            # State transitions (simple + stable)
            prev = rec.state
            new_state = prev

            if rec.stop and rec.current_price is not None and rec.current_price <= rec.stop:
                new_state = SetupState.INVALIDATED.value
                rec.invalid_reason = "Stop broken"
            elif rec.entry and rec.current_price is not None and rec.current_price >= rec.entry:
                new_state = SetupState.ENTRY_TRIGGERED.value
            else:
                # If it exists in scan results, treat as monitoring/forming depending on fields
                if rec.base_high and rec.base_low:
                    new_state = SetupState.MONITORING.value
                else:
                    new_state = SetupState.SETUP_FORMING.value

            if new_state != prev:
                rec.state = new_state
                rec.last_state_change = now

            self.records[sym] = rec

    def prune_missing(self, keep_misses: int = 3):
        for sym in list(self.records.keys()):
            rec = self.records[sym]
            rec.miss_count += 1
            if rec.miss_count >= keep_misses:
                del self.records[sym]

    def to_dataframe(self, exchange: str = "NASDAQ") -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame([])
        rows = []
        for rec in self.records.values():
            d = asdict(rec)
            d["tradingview"] = tradingview_link(rec.symbol, exchange)
            rows.append(d)
        df = pd.DataFrame(rows)
        if "last_state_change" in df.columns:
            df = df.sort_values("last_state_change", ascending=False)
        return df


# -----------------------------
# UltraScannerEngine (dashboard interface)
# -----------------------------
class UltraScannerEngine:
    """
    Backward-compatible engine used by your original dashboard.

    Dashboard expectations:
      scanner = UltraScannerEngine(...)
      results = scanner.run_full_scan()   # no args

    This class will NEVER require a provider to be passed.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        data_provider: Optional[Callable[[str], Optional[Dict[str, List[float]]]]] = None,
        exchange: str = "NASDAQ",
    ):
        self.exchange = exchange
        self.tickers = tickers if tickers is not None else self._default_tickers()
        self.data_provider = data_provider if data_provider is not None else yf_provider

    @staticmethod
    def _default_tickers() -> List[str]:
        try:
            from complete_tickers import get_all_tickers  # type: ignore
            syms = get_all_tickers()
            return list(dict.fromkeys([str(s).upper() for s in syms]))
        except Exception:
            return ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"]

    def set_universe(self, tickers: List[str], data_provider: Optional[Callable[[str], Optional[Dict[str, List[float]]]]] = None):
        self.tickers = list(dict.fromkeys([str(s).upper() for s in tickers]))
        if data_provider is not None:
            self.data_provider = data_provider
        return self

    # --- minimal, stable setup logic ---
    @staticmethod
    def _sma(values: List[float], n: int) -> Optional[float]:
        if len(values) < n:
            return None
        return sum(values[-n:]) / n

    def _detect_setup(self, sym: str, data: Dict[str, List[float]]) -> Optional[Dict[str, Any]]:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        if len(close) < 60:
            return None

        price = float(close[-1])
        sma20 = self._sma(close, 20)
        sma50 = self._sma(close, 50)
        if sma20 is None or sma50 is None:
            return None

        uptrend = price > sma20 > sma50

        # "base" last 7 bars tight range
        base_high = max(high[-7:])
        base_low = min(low[-7:])
        rng = (base_high - base_low) / base_high if base_high else 1.0
        tight = rng <= 0.08

        # entry = base_high breakout, stop = base_low
        entry = float(base_high) if tight else None
        stop = float(base_low) if tight else None

        # score: distance above sma20 + tightness bonus
        score = ((price / sma20) - 1.0) * 100.0
        if tight:
            score += 2.0
        if uptrend:
            score += 2.0

        # filter: require uptrend and either tight base or modest pullback
        if not uptrend:
            return None

        return {
            "symbol": sym,
            "setup_type": "Qullamaggie",
            "score": round(float(score), 2),
            "entry": entry,
            "stop": stop,
            "current_price": round(price, 2),
            "base_high": round(float(base_high), 2),
            "base_low": round(float(base_low), 2),
            "tradingview": tradingview_link(sym, self.exchange),
        }

    def run_full_scan(self) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []
        for sym in self.tickers:
            data = self.data_provider(sym)
            if not data:
                continue
            setup = self._detect_setup(sym, data)
            if setup:
                results.append(setup)

        df = pd.DataFrame(results)
        if not df.empty and "score" in df.columns:
            df = df.sort_values("score", ascending=False)
        return df


# Convenience alias some versions of your app may use
QullamaggieEnhancedScanner = UltraScannerEngine

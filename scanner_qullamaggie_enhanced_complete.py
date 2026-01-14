# scanner_qullamaggie_enhanced_complete.py
# FULL DROP-IN FILE — Scanner State Engine + Monitoring
#
# Exposes (required by ultimate_platform_ENHANCED.py):
#   - SetupState
#   - setup_store
#   - run_scan(symbols, data_provider)
#
# This is Streamlit-safe: no infinite loops. It evaluates symbols once per call.

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional


class SetupState(str, Enum):
    NONE = "NONE"
    SETUP_FORMING = "SETUP_FORMING"
    MONITORING = "MONITORING"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    INVALIDATED = "INVALIDATED"


# Dashboard reads from this store
setup_store: Dict[str, Dict[str, Any]] = {}


def _init_setup(symbol: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "state": SetupState.NONE,
        "base_high": None,
        "base_low": None,
        "entry_price": None,
        "days_in_base": 0,
        "last_price": None,
        "last_state_change": datetime.utcnow(),
        "invalid_reason": None,
    }


# ---------------------------
# QULLAMAGGIE-STYLE HELPERS
# ---------------------------
def _strong_uptrend(data: Dict[str, List[float]]) -> bool:
    # Simple default; replace with your existing enhanced logic if you have it
    c = data["close"]
    if len(c) < 50:
        return False
    return c[-1] > c[-20] and c[-20] > c[-50]


def _pullback_ok(data: Dict[str, List[float]]) -> bool:
    c = data["close"]
    if len(c) < 60:
        return False
    recent_low = min(c[-10:])
    prior = c[-40]
    return recent_low > prior * 0.95  # not a deep breakdown


def _volume_contracting(data: Dict[str, List[float]]) -> bool:
    v = data.get("volume")
    if not v or len(v) < 30:
        return True
    return sum(v[-5:]) < sum(v[-15:-10])  # crude contraction


def _volume_expansion(data: Dict[str, List[float]]) -> bool:
    v = data.get("volume")
    if not v or len(v) < 30:
        return True
    return v[-1] > (sum(v[-20:-1]) / 19) * 1.2


def _detect_base(data: Dict[str, List[float]]) -> Optional[tuple]:
    h = data["high"]
    l = data["low"]
    if len(h) < 20 or len(l) < 20:
        return None
    window_h = h[-7:]
    window_l = l[-7:]
    base_high = max(window_h)
    base_low = min(window_l)
    if base_high <= 0 or base_low <= 0:
        return None
    # Require "tight-ish" range
    rng = (base_high - base_low) / base_high
    if rng > 0.08:
        return None
    return base_high, base_low


def _evaluate_symbol(symbol: str, data: Dict[str, List[float]]) -> Dict[str, Any]:
    if symbol not in setup_store:
        setup_store[symbol] = _init_setup(symbol)

    setup = setup_store[symbol]
    prev = setup["state"]

    price = float(data["close"][-1])
    setup["last_price"] = price

    uptrend = _strong_uptrend(data)
    pullback = _pullback_ok(data)
    base = _detect_base(data)
    vol_contract = _volume_contracting(data)

    # NONE -> SETUP_FORMING
    if setup["state"] == SetupState.NONE:
        if uptrend and pullback:
            setup["state"] = SetupState.SETUP_FORMING

    # SETUP_FORMING -> MONITORING
    elif setup["state"] == SetupState.SETUP_FORMING:
        if base and vol_contract:
            base_high, base_low = base
            setup["state"] = SetupState.MONITORING
            setup["base_high"] = float(base_high)
            setup["base_low"] = float(base_low)
            setup["days_in_base"] = 1

    # MONITORING -> ENTRY / INVALIDATED
    elif setup["state"] == SetupState.MONITORING:
        setup["days_in_base"] = int(setup.get("days_in_base", 0)) + 1

        # Update base if still valid
        if base:
            base_high, base_low = base
            setup["base_high"] = float(base_high)
            setup["base_low"] = float(base_low)

        base_high = setup.get("base_high")
        base_low = setup.get("base_low")

        breakout = False
        if base_high is not None:
            breakout = price > float(base_high) and _volume_expansion(data)

        if breakout:
            setup["state"] = SetupState.ENTRY_TRIGGERED
            setup["entry_price"] = price

        elif base_low is not None and price < float(base_low):
            setup["state"] = SetupState.INVALIDATED
            setup["invalid_reason"] = "Base breakdown"

        # Stale cleanup: too long in base
        if setup["state"] == SetupState.MONITORING and setup["days_in_base"] > 25:
            setup["state"] = SetupState.INVALIDATED
            setup["invalid_reason"] = "Stale (time stop)"

    # ENTRY_TRIGGERED -> leave as-is (exits handled elsewhere)
    # INVALIDATED -> leave as-is (dashboard can filter or you can prune)

    if setup["state"] != prev:
        setup["last_state_change"] = datetime.utcnow()

    return setup


def run_scan(symbols: List[str], data_provider: Callable[[str], Optional[Dict[str, List[float]]]]) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate each symbol once. data_provider(symbol) must return OHLCV dict.
    """
    for sym in symbols:
        try:
            data = data_provider(sym)
            if not data:
                continue
            # Require keys
            if "close" not in data or "high" not in data or "low" not in data:
                continue
            _evaluate_symbol(sym, data)
        except Exception:
            # Never let one ticker break the whole scan
            continue
    return setup_store

#!/usr/bin/env python3
"""
ZeroPoint PRO Backtest Engine v2
=================================

Backtests the ZeroPoint PRO ATR trailing stop flip strategy across multiple
symbols and timeframes using historical MT5 data.

KEY FIX from v1: Now properly exits trades at TP levels instead of holding
until signal flip. Tests THREE exit strategies:
  1. Full exit at TP1 hit
  2. Partial exits: 50% at TP1, 30% at TP2, 20% at TP3/signal flip
  3. Hold until signal flip (original, for comparison)

Strategy Rules (from TradingView Pine Script):
- ATR trailing stop with period=10, multiplier=3.0
- Position flips generate BUY/SELL signals
- Smart Structure SL using swing high/low with ATR minimum
- Three take-profit levels at 2.0x, 3.5x, 5.0x ATR

Usage:
    python zeropoint_backtest.py
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKING_DIR = r"C:\Users\Shadow\.claude-worktrees\e6a2f154-93fc-423b-970f-22d26231afc1\vigilant-varahamihira"
sys.path.insert(0, WORKING_DIR)
sys.path.insert(0, os.path.join(WORKING_DIR, "app"))

SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD",
]

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

HISTORY_DAYS = 180  # 6 months

# Strategy parameters (from Pine Script)
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0
TP1_MULT = 2.0
TP2_MULT = 3.5
TP3_MULT = 5.0
SWING_LOOKBACK = 10
SL_BUFFER_PCT = 0.001  # 0.1%
SL_ATR_MIN_MULT = 1.5

# Partial exit allocation for Strategy 2
TP1_PORTION = 0.50   # Close 50% at TP1
TP2_PORTION = 0.30   # Close 30% at TP2
TP3_PORTION = 0.20   # Close remaining 20% at TP3 or signal flip

# Spread costs (in pips)
SPREAD_PIPS = {
    "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5, "AUDUSD": 2.0,
    "USDCAD": 2.5, "NZDUSD": 2.5, "EURJPY": 2.5, "GBPJPY": 3.0,
    "BTCUSD": 50.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Represents a single completed trade (or partial trade)."""
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    sl_price: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    pnl_pips: float
    portion: float = 1.0   # What fraction of the full position this represents
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    sl_hit: bool = False
    max_profit_pips: float = 0.0
    exit_reason: str = ""

    @property
    def is_win(self) -> bool:
        """Win = positive PnL."""
        return self.pnl_pips > 0


@dataclass
class SymbolStats:
    """Aggregated statistics for one symbol on one timeframe."""
    symbol: str
    timeframe: str
    strategy: str = ""
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def losses(self) -> int:
        return self.total_trades - self.wins

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def weighted_pnl(self) -> float:
        """PnL weighted by position portion (for partial exit strategies)."""
        return sum(t.pnl_pips * t.portion for t in self.trades)

    @property
    def avg_winner_pips(self) -> float:
        winners = [t.pnl_pips for t in self.trades if t.is_win]
        return np.mean(winners) if winners else 0.0

    @property
    def avg_loser_pips(self) -> float:
        losers = [t.pnl_pips for t in self.trades if not t.is_win]
        return np.mean(losers) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pips * t.portion for t in self.trades if t.pnl_pips > 0)
        gross_loss = abs(sum(t.pnl_pips * t.portion for t in self.trades if t.pnl_pips < 0))
        return (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    @property
    def expectancy(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.weighted_pnl / self.total_trades

    @property
    def total_pips(self) -> float:
        return self.weighted_pnl

    @property
    def max_consecutive_wins(self) -> int:
        return self._max_consecutive(is_win=True)

    @property
    def max_consecutive_losses(self) -> int:
        return self._max_consecutive(is_win=False)

    @property
    def max_drawdown_pips(self) -> float:
        if not self.trades:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            equity += t.pnl_pips * t.portion
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        return max_dd

    def _max_consecutive(self, is_win: bool) -> int:
        max_streak = 0
        current = 0
        for t in self.trades:
            if t.is_win == is_win:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak


# ---------------------------------------------------------------------------
# MT5 connection and data fetching
# ---------------------------------------------------------------------------

def normalize_symbol(symbol: str) -> str:
    import re
    return re.sub(r"[^A-Z0-9]", "", symbol.upper())


def resolve_symbol(requested: str) -> Optional[str]:
    norm = normalize_symbol(requested)
    if not norm:
        return None
    symbols = mt5.symbols_get() or []
    names = [s.name for s in symbols if getattr(s, "name", None)]
    if not names:
        return None

    candidates: List[str] = []
    def add(name):
        if name and name not in candidates:
            candidates.append(name)

    add(requested)
    for name in sorted(names, key=len):
        if normalize_symbol(name) == norm:
            add(name)
    for name in sorted(names, key=len):
        if norm in normalize_symbol(name):
            add(name)
    if norm.endswith("USD") and len(norm) > 3:
        base = norm[:-3]
        for name in sorted(names, key=len):
            if normalize_symbol(name).startswith(base):
                add(name)
    for candidate in candidates:
        if not mt5.symbol_select(candidate, True):
            continue
        rates = mt5.copy_rates_from_pos(candidate, mt5.TIMEFRAME_M15, 0, 5)
        if rates is not None and len(rates) >= 5:
            return candidate
    return None


def get_pip_value(symbol: str) -> float:
    norm = normalize_symbol(symbol)
    if norm.endswith("JPY"):
        return 0.01
    if norm.startswith("BTC"):
        return 1.0
    return 0.0001


def get_spread_price(symbol: str) -> float:
    """Return spread cost in price units."""
    pip_val = get_pip_value(symbol)
    norm = normalize_symbol(symbol)
    spread_pips = SPREAD_PIPS.get(norm, 2.0)
    return spread_pips * pip_val


def fetch_data(symbol: str, tf_name: str, tf_const: int, days: int) -> Optional[pd.DataFrame]:
    resolved = resolve_symbol(symbol)
    if not resolved:
        return None
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    rates = mt5.copy_rates_range(resolved, tf_const, start_date, end_date)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df[["open", "high", "low", "close", "tick_volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Pine Script ta.atr() uses Wilder's RMA (alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    atr.iloc[:period - 1] = np.nan
    return atr


# ---------------------------------------------------------------------------
# ZeroPoint PRO strategy core
# ---------------------------------------------------------------------------

def compute_trailing_stop_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ATR trailing stop, position direction, and BUY/SELL signals.
    Matches Pine Script logic exactly.
    """
    atr = compute_atr(df, ATR_PERIOD)
    n_loss = ATR_MULTIPLIER * atr

    close = df["close"].values
    n = len(close)
    n_loss_arr = n_loss.values

    stop = np.zeros(n, dtype=np.float64)
    pos = np.zeros(n, dtype=np.int32)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)

    first_valid = ATR_PERIOD
    if first_valid >= n:
        df["atr"] = atr
        df["nLoss"] = n_loss
        df["xATRTrailingStop"] = np.nan
        df["pos"] = 0
        df["buy_signal"] = False
        df["sell_signal"] = False
        return df

    stop[first_valid] = close[first_valid]
    pos[first_valid] = 1

    for i in range(first_valid + 1, n):
        nl = n_loss_arr[i]
        if np.isnan(nl):
            stop[i] = stop[i - 1]
            pos[i] = pos[i - 1]
            continue

        prev_stop = stop[i - 1]
        prev_close = close[i - 1]
        cur_close = close[i]

        if cur_close > prev_stop and prev_close > prev_stop:
            stop[i] = max(prev_stop, cur_close - nl)
            pos[i] = 1
        elif cur_close < prev_stop and prev_close < prev_stop:
            stop[i] = min(prev_stop, cur_close + nl)
            pos[i] = -1
        elif cur_close > prev_stop:
            stop[i] = cur_close - nl
            pos[i] = 1
        else:
            stop[i] = cur_close + nl
            pos[i] = -1

        if pos[i] == 1 and pos[i - 1] == -1:
            buy_sig[i] = True
        elif pos[i] == -1 and pos[i - 1] == 1:
            sell_sig[i] = True

    df["atr"] = atr
    df["nLoss"] = n_loss
    df["xATRTrailingStop"] = stop
    df["pos"] = pos
    df["buy_signal"] = buy_sig
    df["sell_signal"] = sell_sig
    return df


def compute_smart_sl(df: pd.DataFrame, bar_idx: int, direction: str, atr_val: float) -> float:
    """
    Smart Structure SL from Pine Script.
    BUY: SL = min(recent_swing_low - buffer, close - ATR * 1.5)
    SELL: SL = max(recent_swing_high + buffer, close + ATR * 1.5)
    """
    lookback_start = max(0, bar_idx - SWING_LOOKBACK + 1)
    cur_close = df["close"].iloc[bar_idx]

    if direction == "BUY":
        recent_swing_low = df["low"].iloc[lookback_start:bar_idx + 1].min()
        buffer = recent_swing_low * SL_BUFFER_PCT
        structural_sl = recent_swing_low - buffer
        atr_minimum_sl = cur_close - atr_val * SL_ATR_MIN_MULT
        return min(structural_sl, atr_minimum_sl)
    else:
        recent_swing_high = df["high"].iloc[lookback_start:bar_idx + 1].max()
        buffer = recent_swing_high * SL_BUFFER_PCT
        structural_sl = recent_swing_high + buffer
        atr_maximum_sl = cur_close + atr_val * SL_ATR_MIN_MULT
        return max(structural_sl, atr_maximum_sl)


# ---------------------------------------------------------------------------
# Backtester - Strategy 1: Full exit at TP1
# ---------------------------------------------------------------------------

def backtest_tp1_exit(df: pd.DataFrame, symbol: str, timeframe: str) -> SymbolStats:
    """
    Exit entire position when TP1 is hit.
    This matches how most traders would use the indicator.
    """
    stats = SymbolStats(symbol=symbol, timeframe=timeframe, strategy="TP1 Exit")
    pip_val = get_pip_value(symbol)
    spread = get_spread_price(symbol)

    df = compute_trailing_stop_signals(df)
    n = len(df)

    in_trade = False
    trade_dir = ""
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp3_price = 0.0
    max_profit_pips = 0.0

    for i in range(n):
        atr_val = df["atr"].iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        cur_high = df["high"].iloc[i]
        cur_low = df["low"].iloc[i]
        cur_close = df["close"].iloc[i]
        bar_time = df.index[i]

        if in_trade:
            if trade_dir == "BUY":
                # Check SL first (low touches SL)
                if cur_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, sl_hit=True,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                # Check TP1 hit (high reaches TP1)
                if cur_high >= tp1_price:
                    pnl = (tp1_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp1_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, tp1_hit=True,
                        max_profit_pips=max(max_profit_pips, pnl),
                        exit_reason="TP1 Hit"))
                    in_trade = False
                    continue

                # Track max profit
                bar_profit = (cur_high - entry_price) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)

            else:  # SELL
                if cur_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, sl_hit=True,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                if cur_low <= tp1_price:
                    pnl = (entry_price - tp1_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp1_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, tp1_hit=True,
                        max_profit_pips=max(max_profit_pips, pnl),
                        exit_reason="TP1 Hit"))
                    in_trade = False
                    continue

                bar_profit = (entry_price - cur_low) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)

            # Check opposite signal exit
            exit_opp = (trade_dir == "BUY" and df["sell_signal"].iloc[i]) or \
                       (trade_dir == "SELL" and df["buy_signal"].iloc[i])
            if exit_opp:
                if trade_dir == "BUY":
                    pnl = (cur_close - entry_price) / pip_val
                else:
                    pnl = (entry_price - cur_close) / pip_val
                stats.trades.append(Trade(
                    symbol=symbol, direction=trade_dir,
                    entry_price=entry_price, entry_time=entry_time,
                    exit_price=cur_close, exit_time=bar_time,
                    sl_price=sl_price, tp1_price=tp1_price,
                    tp2_price=tp2_price, tp3_price=tp3_price,
                    pnl_pips=pnl,
                    max_profit_pips=max_profit_pips,
                    exit_reason="Signal Flip"))
                in_trade = False
                # Fall through to open new trade

        # New entry
        if not in_trade:
            if df["buy_signal"].iloc[i]:
                in_trade = True
                trade_dir = "BUY"
                entry_price = cur_close + spread  # Buy at ask
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "BUY", atr_val)
                tp1_price = cur_close + atr_val * TP1_MULT  # TPs from close, not entry
                tp2_price = cur_close + atr_val * TP2_MULT
                tp3_price = cur_close + atr_val * TP3_MULT
                max_profit_pips = 0.0

            elif df["sell_signal"].iloc[i]:
                in_trade = True
                trade_dir = "SELL"
                entry_price = cur_close - spread  # Sell at bid
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "SELL", atr_val)
                tp1_price = cur_close - atr_val * TP1_MULT
                tp2_price = cur_close - atr_val * TP2_MULT
                tp3_price = cur_close - atr_val * TP3_MULT
                max_profit_pips = 0.0

    # Close any open trade
    if in_trade:
        final_close = df["close"].iloc[-1]
        if trade_dir == "BUY":
            pnl = (final_close - entry_price) / pip_val
        else:
            pnl = (entry_price - final_close) / pip_val
        stats.trades.append(Trade(
            symbol=symbol, direction=trade_dir,
            entry_price=entry_price, entry_time=entry_time,
            exit_price=final_close, exit_time=df.index[-1],
            sl_price=sl_price, tp1_price=tp1_price,
            tp2_price=tp2_price, tp3_price=tp3_price,
            pnl_pips=pnl, max_profit_pips=max_profit_pips,
            exit_reason="End of Data"))

    return stats


# ---------------------------------------------------------------------------
# Backtester - Strategy 2: Partial exits at TP levels
# ---------------------------------------------------------------------------

def backtest_partial_tp(df: pd.DataFrame, symbol: str, timeframe: str) -> SymbolStats:
    """
    Partial position exits: 50% at TP1, 30% at TP2, 20% at TP3/signal flip.
    Move SL to breakeven after TP1 hit.
    """
    stats = SymbolStats(symbol=symbol, timeframe=timeframe, strategy="Partial TP")
    pip_val = get_pip_value(symbol)
    spread = get_spread_price(symbol)

    df = compute_trailing_stop_signals(df)
    n = len(df)

    in_trade = False
    trade_dir = ""
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp3_price = 0.0
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    remaining_portion = 1.0
    max_profit_pips = 0.0

    for i in range(n):
        atr_val = df["atr"].iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        cur_high = df["high"].iloc[i]
        cur_low = df["low"].iloc[i]
        cur_close = df["close"].iloc[i]
        bar_time = df.index[i]

        if in_trade:
            if trade_dir == "BUY":
                # Check SL first
                if cur_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=remaining_portion,
                        sl_hit=True, tp1_hit=tp1_hit, tp2_hit=tp2_hit,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                # Check TP levels (partial exits)
                if not tp1_hit and cur_high >= tp1_price:
                    tp1_hit = True
                    pnl = (tp1_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp1_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=TP1_PORTION, tp1_hit=True,
                        max_profit_pips=pnl, exit_reason="TP1 Partial"))
                    remaining_portion -= TP1_PORTION
                    # Move SL to breakeven
                    sl_price = entry_price + spread  # breakeven including spread

                if not tp2_hit and cur_high >= tp2_price:
                    tp2_hit = True
                    pnl = (tp2_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp2_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=TP2_PORTION,
                        tp1_hit=True, tp2_hit=True,
                        max_profit_pips=pnl, exit_reason="TP2 Partial"))
                    remaining_portion -= TP2_PORTION
                    # Move SL to TP1 level
                    sl_price = tp1_price

                if not tp3_hit and cur_high >= tp3_price:
                    tp3_hit = True
                    pnl = (tp3_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp3_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=remaining_portion,
                        tp1_hit=True, tp2_hit=True, tp3_hit=True,
                        max_profit_pips=pnl, exit_reason="TP3 Partial"))
                    in_trade = False
                    continue

                bar_profit = (cur_high - entry_price) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)

            else:  # SELL
                if cur_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=remaining_portion,
                        sl_hit=True, tp1_hit=tp1_hit, tp2_hit=tp2_hit,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                if not tp1_hit and cur_low <= tp1_price:
                    tp1_hit = True
                    pnl = (entry_price - tp1_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp1_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=TP1_PORTION, tp1_hit=True,
                        max_profit_pips=pnl, exit_reason="TP1 Partial"))
                    remaining_portion -= TP1_PORTION
                    sl_price = entry_price - spread

                if not tp2_hit and cur_low <= tp2_price:
                    tp2_hit = True
                    pnl = (entry_price - tp2_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp2_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=TP2_PORTION,
                        tp1_hit=True, tp2_hit=True,
                        max_profit_pips=pnl, exit_reason="TP2 Partial"))
                    remaining_portion -= TP2_PORTION
                    sl_price = tp1_price

                if not tp3_hit and cur_low <= tp3_price:
                    tp3_hit = True
                    pnl = (entry_price - tp3_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=tp3_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, portion=remaining_portion,
                        tp1_hit=True, tp2_hit=True, tp3_hit=True,
                        max_profit_pips=pnl, exit_reason="TP3 Partial"))
                    in_trade = False
                    continue

                bar_profit = (entry_price - cur_low) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)

            # Check opposite signal exit (for remaining portion)
            exit_opp = (trade_dir == "BUY" and df["sell_signal"].iloc[i]) or \
                       (trade_dir == "SELL" and df["buy_signal"].iloc[i])
            if exit_opp:
                if trade_dir == "BUY":
                    pnl = (cur_close - entry_price) / pip_val
                else:
                    pnl = (entry_price - cur_close) / pip_val
                stats.trades.append(Trade(
                    symbol=symbol, direction=trade_dir,
                    entry_price=entry_price, entry_time=entry_time,
                    exit_price=cur_close, exit_time=bar_time,
                    sl_price=sl_price, tp1_price=tp1_price,
                    tp2_price=tp2_price, tp3_price=tp3_price,
                    pnl_pips=pnl, portion=remaining_portion,
                    tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
                    max_profit_pips=max_profit_pips,
                    exit_reason="Signal Flip"))
                in_trade = False

        # New entry
        if not in_trade:
            if df["buy_signal"].iloc[i]:
                in_trade = True
                trade_dir = "BUY"
                entry_price = cur_close + spread
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "BUY", atr_val)
                tp1_price = cur_close + atr_val * TP1_MULT
                tp2_price = cur_close + atr_val * TP2_MULT
                tp3_price = cur_close + atr_val * TP3_MULT
                tp1_hit = tp2_hit = tp3_hit = False
                remaining_portion = 1.0
                max_profit_pips = 0.0

            elif df["sell_signal"].iloc[i]:
                in_trade = True
                trade_dir = "SELL"
                entry_price = cur_close - spread
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "SELL", atr_val)
                tp1_price = cur_close - atr_val * TP1_MULT
                tp2_price = cur_close - atr_val * TP2_MULT
                tp3_price = cur_close - atr_val * TP3_MULT
                tp1_hit = tp2_hit = tp3_hit = False
                remaining_portion = 1.0
                max_profit_pips = 0.0

    # Close remaining position
    if in_trade:
        final_close = df["close"].iloc[-1]
        if trade_dir == "BUY":
            pnl = (final_close - entry_price) / pip_val
        else:
            pnl = (entry_price - final_close) / pip_val
        stats.trades.append(Trade(
            symbol=symbol, direction=trade_dir,
            entry_price=entry_price, entry_time=entry_time,
            exit_price=final_close, exit_time=df.index[-1],
            sl_price=sl_price, tp1_price=tp1_price,
            tp2_price=tp2_price, tp3_price=tp3_price,
            pnl_pips=pnl, portion=remaining_portion,
            tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
            max_profit_pips=max_profit_pips,
            exit_reason="End of Data"))

    return stats


# ---------------------------------------------------------------------------
# Backtester - Strategy 3: Hold until signal flip (original v1 behavior)
# ---------------------------------------------------------------------------

def backtest_signal_flip(df: pd.DataFrame, symbol: str, timeframe: str) -> SymbolStats:
    """
    Hold trade until opposite signal or SL hit. P/L at actual exit price.
    This is the original v1 behavior for comparison.
    """
    stats = SymbolStats(symbol=symbol, timeframe=timeframe, strategy="Signal Flip")
    pip_val = get_pip_value(symbol)
    spread = get_spread_price(symbol)

    df = compute_trailing_stop_signals(df)
    n = len(df)

    in_trade = False
    trade_dir = ""
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    tp3_price = 0.0
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    max_profit_pips = 0.0

    for i in range(n):
        atr_val = df["atr"].iloc[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        cur_high = df["high"].iloc[i]
        cur_low = df["low"].iloc[i]
        cur_close = df["close"].iloc[i]
        bar_time = df.index[i]

        if in_trade:
            if trade_dir == "BUY":
                if cur_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, sl_hit=True,
                        tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                bar_profit = (cur_high - entry_price) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)
                if cur_high >= tp1_price: tp1_hit = True
                if cur_high >= tp2_price: tp2_hit = True
                if cur_high >= tp3_price: tp3_hit = True

            else:  # SELL
                if cur_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_val
                    stats.trades.append(Trade(
                        symbol=symbol, direction=trade_dir,
                        entry_price=entry_price, entry_time=entry_time,
                        exit_price=sl_price, exit_time=bar_time,
                        sl_price=sl_price, tp1_price=tp1_price,
                        tp2_price=tp2_price, tp3_price=tp3_price,
                        pnl_pips=pnl, sl_hit=True,
                        tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
                        max_profit_pips=max_profit_pips,
                        exit_reason="SL Hit"))
                    in_trade = False
                    continue

                bar_profit = (entry_price - cur_low) / pip_val
                max_profit_pips = max(max_profit_pips, bar_profit)
                if cur_low <= tp1_price: tp1_hit = True
                if cur_low <= tp2_price: tp2_hit = True
                if cur_low <= tp3_price: tp3_hit = True

            # Opposite signal exit
            exit_opp = (trade_dir == "BUY" and df["sell_signal"].iloc[i]) or \
                       (trade_dir == "SELL" and df["buy_signal"].iloc[i])
            if exit_opp:
                if trade_dir == "BUY":
                    pnl = (cur_close - entry_price) / pip_val
                else:
                    pnl = (entry_price - cur_close) / pip_val
                stats.trades.append(Trade(
                    symbol=symbol, direction=trade_dir,
                    entry_price=entry_price, entry_time=entry_time,
                    exit_price=cur_close, exit_time=bar_time,
                    sl_price=sl_price, tp1_price=tp1_price,
                    tp2_price=tp2_price, tp3_price=tp3_price,
                    pnl_pips=pnl,
                    tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
                    max_profit_pips=max_profit_pips,
                    exit_reason="Signal Flip"))
                in_trade = False

        if not in_trade:
            if df["buy_signal"].iloc[i]:
                in_trade = True
                trade_dir = "BUY"
                entry_price = cur_close + spread
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "BUY", atr_val)
                tp1_price = cur_close + atr_val * TP1_MULT
                tp2_price = cur_close + atr_val * TP2_MULT
                tp3_price = cur_close + atr_val * TP3_MULT
                tp1_hit = tp2_hit = tp3_hit = False
                max_profit_pips = 0.0

            elif df["sell_signal"].iloc[i]:
                in_trade = True
                trade_dir = "SELL"
                entry_price = cur_close - spread
                entry_time = bar_time
                sl_price = compute_smart_sl(df, i, "SELL", atr_val)
                tp1_price = cur_close - atr_val * TP1_MULT
                tp2_price = cur_close - atr_val * TP2_MULT
                tp3_price = cur_close - atr_val * TP3_MULT
                tp1_hit = tp2_hit = tp3_hit = False
                max_profit_pips = 0.0

    if in_trade:
        final_close = df["close"].iloc[-1]
        if trade_dir == "BUY":
            pnl = (final_close - entry_price) / pip_val
        else:
            pnl = (entry_price - final_close) / pip_val
        stats.trades.append(Trade(
            symbol=symbol, direction=trade_dir,
            entry_price=entry_price, entry_time=entry_time,
            exit_price=final_close, exit_time=df.index[-1],
            sl_price=sl_price, tp1_price=tp1_price,
            tp2_price=tp2_price, tp3_price=tp3_price,
            pnl_pips=pnl,
            tp1_hit=tp1_hit, tp2_hit=tp2_hit, tp3_hit=tp3_hit,
            max_profit_pips=max_profit_pips,
            exit_reason="End of Data"))

    return stats


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    width = 100
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_stats_table(all_stats: List[SymbolStats], timeframe: str, strategy: str) -> None:
    print_header(f"{strategy}  |  {timeframe}  |  {HISTORY_DAYS} days")
    print()

    col_fmt = (
        "{symbol:<10} {trades:>6} {wins:>5} {losses:>6} {winrate:>7} "
        "{pf:>8} {expect:>9} {avg_win:>9} {avg_loss:>10} "
        "{max_dd:>9} {total:>10}"
    )

    header = col_fmt.format(
        symbol="Symbol", trades="Trades", wins="Wins", losses="Losses",
        winrate="Win %", pf="PF", expect="Expect", avg_win="AvgWin",
        avg_loss="AvgLoss", max_dd="MaxDD", total="TotalPips",
    )
    print(header)
    print("-" * len(header))

    tf_stats = [s for s in all_stats if s.timeframe == timeframe and s.strategy == strategy]
    for s in tf_stats:
        pf_str = f"{s.profit_factor:.2f}" if s.profit_factor != float('inf') else "INF"
        print(col_fmt.format(
            symbol=s.symbol,
            trades=s.total_trades,
            wins=s.wins,
            losses=s.losses,
            winrate=f"{s.win_rate:.1f}%",
            pf=pf_str,
            expect=f"{s.expectancy:.1f}",
            avg_win=f"{s.avg_winner_pips:.1f}",
            avg_loss=f"{s.avg_loser_pips:.1f}",
            max_dd=f"{s.max_drawdown_pips:.1f}",
            total=f"{s.total_pips:.1f}",
        ))

    # Aggregate
    all_trades = [t for s in tf_stats for t in s.trades]
    if all_trades:
        total_n = len(all_trades)
        total_wins = sum(1 for t in all_trades if t.is_win)
        total_losses = total_n - total_wins
        total_pips = sum(t.pnl_pips * t.portion for t in all_trades)
        gross_profit = sum(t.pnl_pips * t.portion for t in all_trades if t.pnl_pips > 0)
        gross_loss = abs(sum(t.pnl_pips * t.portion for t in all_trades if t.pnl_pips < 0))
        agg_pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        agg_expect = total_pips / total_n if total_n > 0 else 0
        agg_wr = total_wins / total_n * 100 if total_n > 0 else 0
        avg_w = np.mean([t.pnl_pips for t in all_trades if t.is_win]) if total_wins > 0 else 0
        avg_l = np.mean([t.pnl_pips for t in all_trades if not t.is_win]) if total_losses > 0 else 0

        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(all_trades, key=lambda x: x.entry_time):
            equity += t.pnl_pips * t.portion
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)

        pf_str = f"{agg_pf:.2f}" if agg_pf != float('inf') else "INF"
        print("-" * len(header))
        print(col_fmt.format(
            symbol="AGGREGATE",
            trades=total_n,
            wins=total_wins,
            losses=total_losses,
            winrate=f"{agg_wr:.1f}%",
            pf=pf_str,
            expect=f"{agg_expect:.1f}",
            avg_win=f"{avg_w:.1f}",
            avg_loss=f"{avg_l:.1f}",
            max_dd=f"{max_dd:.1f}",
            total=f"{total_pips:.1f}",
        ))
    print()


def print_exit_breakdown(all_stats: List[SymbolStats], timeframe: str, strategy: str) -> None:
    tf_stats = [s for s in all_stats if s.timeframe == timeframe and s.strategy == strategy]
    all_trades = [t for s in tf_stats for t in s.trades]
    if not all_trades:
        return

    print(f"  Exit Breakdown ({timeframe} - {strategy}):")
    reasons: Dict[str, List[Trade]] = {}
    for t in all_trades:
        reasons.setdefault(t.exit_reason, []).append(t)

    for reason, trades in sorted(reasons.items()):
        count = len(trades)
        avg_pnl = np.mean([t.pnl_pips for t in trades])
        total_weighted = sum(t.pnl_pips * t.portion for t in trades)
        win_count = sum(1 for t in trades if t.is_win)
        wr = win_count / count * 100 if count > 0 else 0
        print(f"    {reason:<20} {count:>5} trades   Avg: {avg_pnl:>8.1f} pips   Weighted Total: {total_weighted:>10.1f}   Win: {wr:.1f}%")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_header("ZeroPoint PRO Backtest Engine v2")
    print()
    print(f"  Strategy:      ATR Trailing Stop Flip (ZeroPoint PRO)")
    print(f"  ATR Period:    {ATR_PERIOD}")
    print(f"  ATR Mult:      {ATR_MULTIPLIER}x")
    print(f"  TP Levels:     {TP1_MULT}x / {TP2_MULT}x / {TP3_MULT}x ATR")
    print(f"  SL:            Smart Structure (swing {SWING_LOOKBACK} bars) + ATR {SL_ATR_MIN_MULT}x minimum")
    print(f"  History:       {HISTORY_DAYS} days")
    print(f"  Symbols:       {', '.join(SYMBOLS)}")
    print(f"  Timeframes:    {', '.join(TIMEFRAMES.keys())}")
    print()
    print(f"  EXIT STRATEGIES TESTED:")
    print(f"    1. TP1 Exit:     Close 100% at TP1 hit (or SL/signal flip)")
    print(f"    2. Partial TP:   50% at TP1, 30% at TP2, 20% at TP3 (SL to BE after TP1)")
    print(f"    3. Signal Flip:  Hold until opposite signal (original v1)")
    print()

    # Connect to MT5
    print("Connecting to MT5...")
    if not mt5.initialize():
        print(f"  ERROR: MT5 initialization failed: {mt5.last_error()}")
        return

    account = mt5.account_info()
    if account:
        print(f"  Connected: Account {account.login} on {account.server}")
        print(f"  Balance: ${account.balance:.2f}")
    print()

    # Fetch data once per symbol/timeframe, run all 3 strategies
    all_stats: List[SymbolStats] = []
    data_cache: Dict[str, pd.DataFrame] = {}

    for tf_name, tf_const in TIMEFRAMES.items():
        print(f"--- Fetching {tf_name} data ---")
        for symbol in SYMBOLS:
            cache_key = f"{symbol}_{tf_name}"
            print(f"  {symbol} ({tf_name})...", end=" ", flush=True)
            df = fetch_data(symbol, tf_name, tf_const, HISTORY_DAYS)
            if df is None:
                print("SKIPPED (no data)")
                continue
            data_cache[cache_key] = df
            print(f"{len(df)} bars...", end=" ", flush=True)

            # Run all 3 strategies
            s1 = backtest_tp1_exit(df.copy(), symbol, tf_name)
            s2 = backtest_partial_tp(df.copy(), symbol, tf_name)
            s3 = backtest_signal_flip(df.copy(), symbol, tf_name)

            all_stats.extend([s1, s2, s3])
            print(f"TP1:{s1.total_trades}t/{s1.win_rate:.0f}%  Partial:{s2.total_trades}t/{s2.win_rate:.0f}%  Flip:{s3.total_trades}t/{s3.win_rate:.0f}%")

        print()

    # Print detailed results per strategy per timeframe
    strategies = ["TP1 Exit", "Partial TP", "Signal Flip"]
    for tf_name in TIMEFRAMES:
        for strat in strategies:
            print_stats_table(all_stats, tf_name, strat)
            print_exit_breakdown(all_stats, tf_name, strat)

    # =========================================================================
    # GRAND COMPARISON TABLE
    # =========================================================================
    print_header("STRATEGY COMPARISON SUMMARY")
    print()
    print(f"  {'Strategy':<15} {'TF':<5} {'Trades':>6} {'Win%':>7} {'PF':>8} {'Expect':>9} {'Total Pips':>12} {'MaxDD':>10}")
    print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*9} {'-'*12} {'-'*10}")

    for strat in strategies:
        for tf_name in TIMEFRAMES:
            tf_trades = [t for s in all_stats if s.timeframe == tf_name and s.strategy == strat for t in s.trades]
            if not tf_trades:
                continue
            n = len(tf_trades)
            wins = sum(1 for t in tf_trades if t.is_win)
            wr = wins / n * 100
            total = sum(t.pnl_pips * t.portion for t in tf_trades)
            gp = sum(t.pnl_pips * t.portion for t in tf_trades if t.pnl_pips > 0)
            gl = abs(sum(t.pnl_pips * t.portion for t in tf_trades if t.pnl_pips < 0))
            pf = gp / gl if gl > 0 else float('inf') if gp > 0 else 0
            expect = total / n
            # Max DD
            eq = 0.0
            pk = 0.0
            mdd = 0.0
            for t in sorted(tf_trades, key=lambda x: x.entry_time):
                eq += t.pnl_pips * t.portion
                pk = max(pk, eq)
                mdd = max(mdd, pk - eq)

            pf_str = f"{pf:.2f}" if pf != float('inf') else "INF"
            marker = " <-- BEST" if pf > 1.5 and total > 0 else ""
            print(f"  {strat:<15} {tf_name:<5} {n:>6} {wr:>6.1f}% {pf_str:>8} {expect:>8.1f}p {total:>11.1f}p {mdd:>9.1f}p{marker}")
        print()

    # =========================================================================
    # BEST SYMBOL BREAKDOWN
    # =========================================================================
    print_header("BEST PERFORMING SYMBOLS (PF > 1.0)")
    print()
    print(f"  {'Symbol':<10} {'Strategy':<15} {'TF':<5} {'Trades':>6} {'Win%':>7} {'PF':>8} {'Total Pips':>12}")
    print(f"  {'-'*10} {'-'*15} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*12}")

    results = []
    for s in all_stats:
        if s.total_trades >= 5 and s.profit_factor > 1.0:
            results.append(s)

    results.sort(key=lambda x: x.profit_factor, reverse=True)
    for s in results[:30]:
        pf_str = f"{s.profit_factor:.2f}" if s.profit_factor != float('inf') else "INF"
        print(f"  {s.symbol:<10} {s.strategy:<15} {s.timeframe:<5} {s.total_trades:>6} {s.win_rate:>6.1f}% {pf_str:>8} {s.total_pips:>11.1f}p")

    print()

    # Cleanup
    mt5.shutdown()
    print("MT5 disconnected. Backtest complete.")
    print()


if __name__ == "__main__":
    main()

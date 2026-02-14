#!/usr/bin/env python3
"""
Engine-level 1-year replay using MT5 historical candles.

This script replays app/trading_engine.py signal logic (including MTF and
pattern filters) and estimates profitability distributions from a $200 account.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from app.model_manager import NeuralModelManager
from app.trading_engine import TradingEngine, TradingSignal


REQUESTED_SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "BTCUSD",
]

TF_NAME_TO_CONST = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

HORIZON_M15_BARS = 16
HORIZON_SECONDS = HORIZON_M15_BARS * 15 * 60
MIN_SIGNAL_BARS = 120


@dataclass
class SymbolMeta:
    requested: str
    resolved: str
    digits: int
    point: float
    spread_points: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float


@dataclass
class OpenTrade:
    symbol: str
    action: str
    entry_time: int
    close_time: int
    entry_balance: float
    pnl: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_epoch_seconds(dt: datetime) -> int:
    return int(dt.timestamp())


def _normalize_symbol(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def resolve_symbol(requested: str, all_names: List[str]) -> Optional[str]:
    requested_u = requested.upper()
    requested_norm = _normalize_symbol(requested)
    normalized_map = {name: _normalize_symbol(name) for name in all_names}

    exact = [name for name in all_names if name.upper() == requested_u]
    if exact:
        return exact[0]

    exact_norm = [name for name in all_names if normalized_map[name] == requested_norm]
    if exact_norm:
        exact_norm.sort(key=len)
        return exact_norm[0]

    contains = [name for name in all_names if requested_norm in normalized_map[name]]
    if contains:
        contains.sort(key=lambda n: (0 if normalized_map[n].startswith(requested_norm) else 1, len(n)))
        return contains[0]

    if requested_norm.endswith("USD") and len(requested_norm) > 3:
        base = requested_norm[:-3]
        usd_variants = [
            name for name in all_names
            if normalized_map[name].startswith(base) and "USD" in normalized_map[name]
        ]
        if usd_variants:
            usd_variants.sort(key=len)
            return usd_variants[0]

        base_only = [name for name in all_names if normalized_map[name] == base]
        if base_only:
            base_only.sort(key=len)
            return base_only[0]

    return None


def copy_rates_df(symbol: str, timeframe: int, start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    if "time" not in df.columns:
        return None
    df = df.sort_values("time").reset_index(drop=True)
    return df


def week_start_utc(epoch_s: int) -> datetime:
    dt = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    monday = dt - timedelta(days=dt.weekday())
    return datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc)


class ReplayConnector:
    def __init__(self, symbol_data: Dict[str, Dict[str, pd.DataFrame]], symbol_meta: Dict[str, SymbolMeta], start_balance: float):
        self.symbol_data = symbol_data
        self.symbol_meta = symbol_meta
        self.cursor_time: Dict[str, int] = {}
        self.balance = float(start_balance)

    def set_cursor(self, symbol: str, epoch_s: int) -> None:
        self.cursor_time[symbol] = int(epoch_s)

    def set_balance(self, balance: float) -> None:
        self.balance = float(balance)

    def get_account_info(self) -> Dict[str, float]:
        return {
            "balance": float(self.balance),
            "equity": float(self.balance),
            "margin": 0.0,
            "margin_free": float(self.balance),
            "currency": "USD",
            "leverage": 100,
            "trade_allowed": True,
            "trade_expert": True,
        }

    def get_rates(self, symbol: str, timeframe: int, start_pos: int = 0, count: int = 100) -> Optional[List[Dict[str, float]]]:
        tf_name = None
        for name, const in TF_NAME_TO_CONST.items():
            if const == timeframe:
                tf_name = name
                break
        if tf_name is None:
            return None

        if symbol not in self.symbol_data:
            return None
        df = self.symbol_data[symbol].get(tf_name)
        if df is None or df.empty:
            return None

        cursor = self.cursor_time.get(symbol)
        if cursor is None:
            return None
        window = df.loc[df["time"] <= cursor].tail(int(count))
        if window.empty:
            return None

        cols = ["time", "open", "high", "low", "close", "tick_volume"]
        return window[cols].to_dict("records")

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, float]]:
        if symbol not in self.symbol_data or symbol not in self.symbol_meta:
            return None
        m5 = self.symbol_data[symbol].get("M5")
        if m5 is None or m5.empty:
            return None
        cursor = self.cursor_time.get(symbol)
        if cursor is None:
            return None
        window = m5.loc[m5["time"] <= cursor]
        if window.empty:
            return None
        row = window.iloc[-1]
        meta = self.symbol_meta[symbol]
        mid = float(row["close"])
        spread_points, spread_price = self._synthetic_spread(symbol=symbol, mid=mid, meta=meta)
        ask = mid + spread_price * 0.5
        bid = mid - spread_price * 0.5
        return {
            "requested_name": meta.requested,
            "name": meta.resolved,
            "bid": bid,
            "ask": ask,
            "spread": spread_points,
            "digits": meta.digits,
            "point": meta.point,
            "trade_tick_value": meta.tick_value,
            "trade_tick_size": meta.tick_size,
            "trade_contract_size": meta.contract_size,
            "volume_min": meta.volume_min,
            "volume_max": meta.volume_max,
            "volume_step": meta.volume_step,
            "margin_initial": 0.0,
            "margin_maintenance": 0.0,
            "session_deals": 0,
            "session_buy_orders": 0,
            "session_sell_orders": 0,
            "volume": 0.0,
            "high": float(row["high"]),
            "low": float(row["low"]),
        }

    def _synthetic_spread(self, symbol: str, mid: float, meta: SymbolMeta) -> Tuple[float, float]:
        """
        Use replay-stable spread assumptions instead of live snapshot spread.
        Live spread can be market-closed-wide and would suppress all historical trades.
        """
        normalized = _normalize_symbol(symbol)
        point = max(float(meta.point), 1e-8)

        is_crypto = any(token in normalized for token in ("BTC", "ETH", "XRP", "LTC", "SOL"))
        fx_ccy = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"}
        is_forex = (
            not is_crypto
            and len(normalized) >= 6
            and normalized[:3].isalpha()
            and normalized[3:6].isalpha()
            and normalized[:3] in fx_ccy
            and normalized[3:6] in fx_ccy
        )

        if is_forex:
            quote = normalized[3:6]
            pip_size = 0.01 if quote == "JPY" else 0.0001
            target_pips = 1.4 if quote == "JPY" else 1.1
            spread_price = max(target_pips * pip_size, point)
        elif is_crypto:
            spread_price = max(mid * 0.0012, point * 10.0)
        else:
            spread_price = max(mid * 0.0008, point * 5.0)

        spread_points = float(max(1.0, spread_price / point))
        return spread_points, float(spread_price)

    def place_order(self, order_request: Dict) -> Optional[Dict]:
        return None


def evaluate_trade_outcome(
    signal: TradingSignal,
    symbol_meta: SymbolMeta,
    m5_df: pd.DataFrame,
    entry_time: int,
) -> Optional[Tuple[int, float]]:
    window = m5_df.loc[(m5_df["time"] > entry_time) & (m5_df["time"] <= (entry_time + HORIZON_SECONDS))]
    if window.empty:
        return None

    action = signal.action.upper()
    entry = float(signal.entry_price)
    stop = float(signal.stop_loss)
    tp = float(signal.take_profit)
    vol = float(signal.position_size)

    exit_price = None
    exit_time = None

    for _, bar in window.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        t = int(bar["time"])
        if action == "BUY":
            hit_sl = low <= stop
            hit_tp = high >= tp
            if hit_sl and hit_tp:
                exit_price = stop
                exit_time = t
                break
            if hit_sl:
                exit_price = stop
                exit_time = t
                break
            if hit_tp:
                exit_price = tp
                exit_time = t
                break
        else:
            hit_sl = high >= stop
            hit_tp = low <= tp
            if hit_sl and hit_tp:
                exit_price = stop
                exit_time = t
                break
            if hit_sl:
                exit_price = stop
                exit_time = t
                break
            if hit_tp:
                exit_price = tp
                exit_time = t
                break

    if exit_price is None:
        last_row = window.iloc[-1]
        exit_price = float(last_row["close"])
        exit_time = int(last_row["time"])

    tick_size = symbol_meta.tick_size if symbol_meta.tick_size > 0 else max(symbol_meta.point, 1e-6)
    tick_value = symbol_meta.tick_value
    if tick_value <= 0:
        tick_value = max(symbol_meta.contract_size * tick_size, 1e-6)

    if action == "BUY":
        price_diff = exit_price - entry
    else:
        price_diff = entry - exit_price

    pnl = (price_diff / tick_size) * tick_value * vol
    return exit_time, float(pnl)


def calc_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def monte_carlo_annual_from_weeks(weekly_returns: List[float], simulations: int = 5000, weeks: int = 52) -> Dict[str, float]:
    if not weekly_returns:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(weekly_returns, dtype=float)
    sims = np.empty(simulations, dtype=float)
    rng = np.random.default_rng(42)
    for i in range(simulations):
        draw = rng.choice(arr, size=weeks, replace=True)
        sims[i] = float(np.prod(1.0 + draw) - 1.0)
    return {
        "p10": float(np.percentile(sims, 10)),
        "p50": float(np.percentile(sims, 50)),
        "p90": float(np.percentile(sims, 90)),
    }


def run_scenario(
    symbols: List[str],
    connector: ReplayConnector,
    model_manager: NeuralModelManager,
    start_balance: float,
    risk_per_trade: float,
    profitability_first_mode: bool,
    confidence_threshold: float = 0.35,
    replay_start_epoch: Optional[int] = None,
    replay_end_epoch: Optional[int] = None,
) -> Dict[str, float]:
    engine = TradingEngine(
        mt5_connector=connector,
        model_manager=model_manager,
        risk_per_trade=float(risk_per_trade),
        confidence_threshold=float(confidence_threshold),
        trading_pairs=list(symbols),
        max_concurrent_positions=5,
    )
    engine.immediate_trade_mode = True
    engine.profitability_first_mode = bool(profitability_first_mode)
    engine.model_pattern_conflict_block = False
    engine.mtf_alignment_enabled = False
    engine.tail_risk_control_enabled = False
    engine.model_min_trade_score = min(engine.model_min_trade_score, 0.33)
    engine.model_min_directional_gap = min(engine.model_min_directional_gap, 0.005)
    engine.symbol_entry_cooldown_seconds = min(engine.symbol_entry_cooldown_seconds, 60)
    engine.max_new_trades_per_hour = max(engine.max_new_trades_per_hour, len(symbols) * 2)
    engine._rebuild_live_symbol_profile()

    # Build M15 event timeline.
    m15_times_by_symbol: Dict[str, np.ndarray] = {}
    timeline_points: List[np.ndarray] = []
    for sym in symbols:
        df = connector.symbol_data[sym]["M15"]
        times = df["time"].values.astype(np.int64)
        # keep only points with enough history and enough future bars
        if len(times) <= (MIN_SIGNAL_BARS + HORIZON_M15_BARS):
            continue
        valid = times[MIN_SIGNAL_BARS : len(times) - HORIZON_M15_BARS]
        m15_times_by_symbol[sym] = valid
        timeline_points.append(valid)

    if not timeline_points:
        return {
            "samples": 0,
            "trades": 0,
            "trade_rate": 0.0,
            "realized_return_1y": 0.0,
            "realized_end_balance_from_start": start_balance,
            "weekly_p10": 0.0,
            "weekly_p50": 0.0,
            "weekly_p90": 0.0,
            "weekly_prob_positive": 0.0,
            "annual_p10": 0.0,
            "annual_p50": 0.0,
            "annual_p90": 0.0,
        }

    all_times = np.unique(np.concatenate(timeline_points))
    all_times.sort()
    if replay_start_epoch is not None:
        all_times = all_times[all_times >= int(replay_start_epoch)]
    if replay_end_epoch is not None:
        all_times = all_times[all_times <= int(replay_end_epoch)]
    m15_time_sets = {s: set(v.tolist()) for s, v in m15_times_by_symbol.items()}
    if all_times.size == 0:
        return {
            "samples": 0,
            "trades": 0,
            "trade_rate": 0.0,
            "realized_return_1y": 0.0,
            "realized_end_balance_from_start": start_balance,
            "weekly_p10": 0.0,
            "weekly_p50": 0.0,
            "weekly_p90": 0.0,
            "weekly_prob_positive": 0.0,
            "annual_p10": 0.0,
            "annual_p50": 0.0,
            "annual_p90": 0.0,
        }

    balance = float(start_balance)
    open_trades: List[OpenTrade] = []
    open_symbols: set[str] = set()
    last_entry_by_symbol: Dict[str, int] = {}
    hourly_entry_times: List[int] = []

    weekly_factors: Dict[datetime, float] = defaultdict(lambda: 1.0)
    trade_returns_pct: List[float] = []
    samples = 0
    trades = 0

    def close_due(now_s: int) -> None:
        nonlocal balance, open_trades, open_symbols
        if not open_trades:
            return
        open_trades.sort(key=lambda x: x.close_time)
        still_open: List[OpenTrade] = []
        for tr in open_trades:
            if tr.close_time <= now_s:
                pre_balance = balance
                balance = max(0.01, balance + tr.pnl)
                ret = tr.pnl / tr.entry_balance if tr.entry_balance > 0 else 0.0
                trade_returns_pct.append(float(ret))
                w = week_start_utc(tr.close_time)
                weekly_factors[w] *= (1.0 + float(ret))
                open_symbols.discard(tr.symbol)
                if not math.isfinite(pre_balance):
                    raise RuntimeError("Invalid balance state")
            else:
                still_open.append(tr)
        open_trades = still_open

    for t in all_times.tolist():
        t = int(t)
        close_due(t)

        cutoff = t - 3600
        hourly_entry_times = [x for x in hourly_entry_times if x >= cutoff]

        for sym in symbols:
            if sym not in m15_time_sets:
                continue
            if t not in m15_time_sets[sym]:
                continue

            samples += 1
            connector.set_balance(balance)
            connector.set_cursor(sym, t)

            try:
                signal = engine._generate_signal(sym)
            except Exception:
                signal = None

            if signal is None or signal.action.upper() not in ("BUY", "SELL"):
                continue

            last_entry = last_entry_by_symbol.get(sym)
            if last_entry is not None and (t - last_entry) < int(engine.symbol_entry_cooldown_seconds):
                continue
            if len(hourly_entry_times) >= int(engine.max_new_trades_per_hour):
                continue
            if len(open_trades) >= int(engine.max_concurrent_positions):
                continue
            if sym in open_symbols:
                continue

            outcome = evaluate_trade_outcome(
                signal=signal,
                symbol_meta=connector.symbol_meta[sym],
                m5_df=connector.symbol_data[sym]["M5"],
                entry_time=t,
            )
            if outcome is None:
                continue

            close_time, pnl = outcome
            trade = OpenTrade(
                symbol=sym,
                action=signal.action.upper(),
                entry_time=t,
                close_time=int(close_time),
                entry_balance=float(balance),
                pnl=float(pnl),
            )
            open_trades.append(trade)
            open_symbols.add(sym)
            last_entry_by_symbol[sym] = t
            hourly_entry_times.append(t)
            trades += 1

    if all_times.size > 0:
        close_due(int(all_times[-1]) + HORIZON_SECONDS + 1)

    weekly_returns = [factor - 1.0 for _, factor in sorted(weekly_factors.items(), key=lambda kv: kv[0])]
    weekly_stats = calc_percentiles(weekly_returns)
    annual_stats = monte_carlo_annual_from_weeks(weekly_returns, simulations=5000, weeks=52)

    realized_return = (balance / float(start_balance)) - 1.0
    trade_rate = float(trades / samples) if samples > 0 else 0.0
    weekly_prob_positive = float(np.mean(np.array(weekly_returns, dtype=float) > 0.0)) if weekly_returns else 0.0

    return {
        "samples": int(samples),
        "trades": int(trades),
        "trade_rate": float(trade_rate),
        "realized_return_1y": float(realized_return),
        "realized_end_balance_from_start": float(balance),
        "weekly_p10": float(weekly_stats["p10"]),
        "weekly_p50": float(weekly_stats["p50"]),
        "weekly_p90": float(weekly_stats["p90"]),
        "weekly_prob_positive": float(weekly_prob_positive),
        "annual_p10": float(annual_stats["p10"]),
        "annual_p50": float(annual_stats["p50"]),
        "annual_p90": float(annual_stats["p90"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 1-year engine replay with MT5 candles.")
    parser.add_argument("--model", default="neural_model.pth", help="Path to model .pth")
    parser.add_argument("--start-balance", type=float, default=200.0, help="Starting account balance")
    parser.add_argument("--risk-per-trade", type=float, default=0.05, help="Risk per trade (e.g. 0.05 for 5%)")
    parser.add_argument("--days", type=int, default=365, help="Historical days to replay")
    parser.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="Optional history days to fetch (uses max(days, 14) when omitted).",
    )
    parser.add_argument(
        "--scenario",
        choices=["both", "all", "enabled"],
        default="both",
        help="Which replay scenario(s) to run",
    )
    parser.add_argument("--output", default="yearly_engine_replay.json", help="Output JSON path")
    args = parser.parse_args()

    if not mt5.initialize():
        print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
        return 1

    model_manager = NeuralModelManager()
    if not model_manager.load_model(args.model):
        print(f"ERROR: Could not load model from {args.model}")
        mt5.shutdown()
        return 1

    symbols_all = mt5.symbols_get() or []
    all_names = [s.name for s in symbols_all if getattr(s, "name", None)]

    def build_symbol_bundle(days: int, end_dt: datetime) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, SymbolMeta], Dict[str, str]]:
        start_dt = end_dt - timedelta(days=int(days))
        bundle_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        bundle_meta: Dict[str, SymbolMeta] = {}
        bundle_resolved: Dict[str, str] = {}

        for req in REQUESTED_SYMBOLS:
            resolved = resolve_symbol(req, all_names)
            if not resolved:
                continue
            info = mt5.symbol_info(resolved)
            if info is None:
                continue

            data_per_tf: Dict[str, pd.DataFrame] = {}
            tf_ok = True
            for tf_name, tf_const in TF_NAME_TO_CONST.items():
                df = copy_rates_df(resolved, tf_const, start_dt, end_dt)
                if df is None or len(df) < (MIN_SIGNAL_BARS + HORIZON_M15_BARS + 10):
                    tf_ok = False
                    break
                data_per_tf[tf_name] = df
            if not tf_ok:
                continue

            bundle_data[req] = data_per_tf
            bundle_meta[req] = SymbolMeta(
                requested=req,
                resolved=resolved,
                digits=int(getattr(info, "digits", 5) or 5),
                point=float(getattr(info, "point", 0.0001) or 0.0001),
                spread_points=float(getattr(info, "spread", 20.0) or 20.0),
                tick_size=float(getattr(info, "trade_tick_size", 0.0) or 0.0),
                tick_value=float(getattr(info, "trade_tick_value", 0.0) or 0.0),
                contract_size=float(getattr(info, "trade_contract_size", 1.0) or 1.0),
                volume_min=float(getattr(info, "volume_min", 0.01) or 0.01),
                volume_max=float(getattr(info, "volume_max", 100.0) or 100.0),
                volume_step=float(getattr(info, "volume_step", 0.01) or 0.01),
            )
            bundle_resolved[req] = resolved

        return bundle_data, bundle_meta, bundle_resolved

    replay_end_dt = _utc_now()
    requested_days = int(args.days)
    history_days = int(args.history_days) if args.history_days is not None else max(requested_days, 14)
    effective_history_days = history_days
    symbol_data, symbol_meta, resolved_summary = build_symbol_bundle(effective_history_days, replay_end_dt)
    if not symbol_data:
        fallback_days = [180, 150, 120, 90, 60, 30]
        for d in fallback_days:
            if d >= history_days:
                continue
            symbol_data, symbol_meta, resolved_summary = build_symbol_bundle(d, replay_end_dt)
            if symbol_data:
                effective_history_days = d
                print(
                    f"INFO: Requested {history_days} history days but M5 history was insufficient; "
                    f"auto-falling back to {effective_history_days}d."
                )
                break
    if not symbol_data:
        print("ERROR: No symbols had enough MT5 history for replay.")
        mt5.shutdown()
        return 1

    connector = ReplayConnector(symbol_data=symbol_data, symbol_meta=symbol_meta, start_balance=float(args.start_balance))
    symbols_available = sorted(symbol_data.keys())
    replay_start_dt = replay_end_dt - timedelta(days=requested_days)
    replay_start_epoch = _to_epoch_seconds(replay_start_dt)
    replay_end_epoch = _to_epoch_seconds(replay_end_dt)

    # Determine enabled symbols with profitability-first profile.
    probe_engine = TradingEngine(
        mt5_connector=connector,
        model_manager=model_manager,
        risk_per_trade=float(args.risk_per_trade),
        confidence_threshold=0.65,
        trading_pairs=list(symbols_available),
        max_concurrent_positions=5,
    )
    probe_engine.immediate_trade_mode = False
    probe_engine.profitability_first_mode = True
    probe_engine._rebuild_live_symbol_profile()
    enabled_symbols = [s for s in symbols_available if probe_engine._is_symbol_live_enabled(s)]
    if not enabled_symbols:
        enabled_symbols = symbols_available[:]

    all_symbols_result: Optional[Dict[str, float]] = None
    enabled_only_result: Optional[Dict[str, float]] = None

    if args.scenario in ("both", "all"):
        all_symbols_result = run_scenario(
            symbols=symbols_available,
            connector=connector,
            model_manager=model_manager,
            start_balance=float(args.start_balance),
            risk_per_trade=float(args.risk_per_trade),
            profitability_first_mode=False,
            confidence_threshold=0.35,
            replay_start_epoch=replay_start_epoch,
            replay_end_epoch=replay_end_epoch,
        )
    if args.scenario in ("both", "enabled"):
        enabled_only_result = run_scenario(
            symbols=enabled_symbols,
            connector=connector,
            model_manager=model_manager,
            start_balance=float(args.start_balance),
            risk_per_trade=float(args.risk_per_trade),
            profitability_first_mode=True,
            confidence_threshold=0.35,
            replay_start_epoch=replay_start_epoch,
            replay_end_epoch=replay_end_epoch,
        )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model)),
        "requested_days": int(args.days),
        "history_days_requested": int(history_days),
        "history_days_effective": int(effective_history_days),
        "start_balance": float(args.start_balance),
        "risk_per_trade": float(args.risk_per_trade),
        "scenario": str(args.scenario),
        "requested_symbols": REQUESTED_SYMBOLS,
        "resolved_symbols": resolved_summary,
        "symbols_with_data": symbols_available,
        "enabled_symbols_from_model_profile": enabled_symbols,
        "all_symbols_scenario": all_symbols_result,
        "enabled_only_scenario": enabled_only_result,
        "hit_annual_p90_target_all_symbols": bool(
            all_symbols_result is not None
            and all_symbols_result["realized_return_1y"] >= all_symbols_result["annual_p90"]
        ),
        "hit_annual_p90_target_enabled_only": bool(
            enabled_only_result is not None
            and enabled_only_result["realized_return_1y"] >= enabled_only_result["annual_p90"]
        ),
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    mt5.shutdown()

    print(f"Replay complete. Output written to {out_path}")
    summary = {
        "requested_days": output["requested_days"],
        "history_days_effective": output["history_days_effective"],
    }
    if output["all_symbols_scenario"] is not None:
        summary.update(
            {
                "all_symbols_realized_return_1y": output["all_symbols_scenario"]["realized_return_1y"],
                "all_symbols_annual_p90": output["all_symbols_scenario"]["annual_p90"],
            }
        )
    if output["enabled_only_scenario"] is not None:
        summary.update(
            {
                "enabled_realized_return_1y": output["enabled_only_scenario"]["realized_return_1y"],
                "enabled_annual_p90": output["enabled_only_scenario"]["annual_p90"],
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

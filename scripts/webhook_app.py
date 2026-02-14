#!/usr/bin/env python3
"""
ACi -- Standalone ZeroPoint Trading App
Fetches candles from MT5, runs ZeroPoint signal detection in Python,
displays live charts with lightweight-charts, and auto-trades on MT5.
No TradingView needed.
"""

import sys, os, threading, json, time as _time, logging
import traceback as _tb
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so "from app.X" / "from models.X" etc. work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QLineEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QFrame, QAbstractItemView, QComboBox,
    QTabWidget, QSplitter, QDialog,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor

import MetaTrader5 as mt5
from lightweight_charts.widgets import QtChart

from app.zeropoint_signal import compute_zeropoint_state, ATR_PERIOD, ATR_MULTIPLIER
from app.zeropoint_signal import TP1_MULT, TP2_MULT, TP3_MULT  # baseline (unused)
from app.zeropoint_signal import (
    TP1_MULT_AGG, TP2_MULT_AGG, TP3_MULT_AGG,       # V4 optimized TPs
    BE_TRIGGER_MULT, BE_BUFFER_MULT,                   # Early breakeven
    PROFIT_TRAIL_DISTANCE_MULT,                        # Post-TP1 trailing
    STALL_BARS,                                        # Stall exit
    MICRO_TP_MULT, MICRO_TP_PCT,                       # Micro-partial
)

# ---------------------------------------------------------------------------
# Crash handler
# ---------------------------------------------------------------------------
def _global_exception_hook(exc_type, exc_value, exc_tb):
    crash_msg = "".join(_tb.format_exception(exc_type, exc_value, exc_tb))
    try:
        with open("crash.log", "a") as f:
            f.write(f"\n{'='*60}\n{datetime.now()}\n{crash_msg}\n")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)
sys.excepthook = _global_exception_hook

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "aci.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("aci")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "GBPJPY"]

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}
TIMEFRAMES = list(TF_MAP.keys())

MAGIC_NUMBER = 234567
SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "aci_settings.json")

# V4 partial close splits (1/3 at TP1, 1/3 at TP2, close rest at TP3)
TP1_CLOSE_PCT = 0.33
TP2_CLOSE_PCT = 0.33   # of original
TP3_CLOSE_PCT = 1.00   # close remaining
MICRO_CLOSE_PCT = MICRO_TP_PCT  # 15% micro-partial at 0.8x ATR

# Margin safety — require at least this % free margin after trade
MIN_MARGIN_LEVEL_PCT = 150   # 150% margin level = safe zone

# ---------------------------------------------------------------------------
# Stylesheets (from trading_app.py)
# ---------------------------------------------------------------------------
CLAUDE_STYLE = """
QMainWindow, QWidget {
    background-color: #F4F3EE;
    color: #141413;
    font-family: Georgia, Cambria, 'Times New Roman', serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #D5D3C8;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 16px;
    font-weight: bold;
    color: #3D3929;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}
QPushButton {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    padding: 8px 20px;
    color: #3D3929;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #E4E2D8;
    border-color: #C15F3C;
}
QPushButton:pressed {
    background-color: #D5D3C8;
}
QPushButton:disabled {
    background-color: #F4F3EE;
    color: #B0AEA5;
    border-color: #E8E6DC;
}
QPushButton#startBtn {
    background-color: #C15F3C;
    border: none;
    color: #FFFFFF;
}
QPushButton#startBtn:hover {
    background-color: #B5523C;
}
QPushButton#startBtn:disabled {
    background-color: #D5D3C8;
    color: #B0AEA5;
}
QPushButton#stopBtn {
    background-color: #E8E6DC;
    border: 1px solid #D5D3C8;
    color: #3D3929;
}
QPushButton#stopBtn:hover {
    background-color: #D5D3C8;
}
QPushButton#tradeBtn {
    background-color: #EEECE2;
    border: 2px solid #C44444;
    color: #C44444;
}
QPushButton#tradeBtn:hover {
    background-color: #E4E2D8;
}
QPushButton#emergencyBtn {
    background-color: #C44444;
    border: none;
    color: #FFFFFF;
}
QPushButton#emergencyBtn:hover {
    background-color: #A83838;
}
QLineEdit {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    padding: 6px 10px;
    color: #141413;
}
QLineEdit:focus {
    border-color: #C15F3C;
}
QCheckBox {
    spacing: 6px;
    color: #141413;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #B0AEA5;
    border-radius: 4px;
    background-color: #FAF9F5;
}
QCheckBox::indicator:checked {
    background-color: #C15F3C;
    border-color: #B5523C;
}
QTableWidget {
    background-color: #FAF9F5;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    gridline-color: #E8E6DC;
    selection-background-color: #E8D5C8;
    color: #141413;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #EEECE2;
    color: #3D3929;
    border: none;
    border-bottom: 1px solid #D5D3C8;
    padding: 6px;
    font-weight: bold;
}
QTextEdit {
    background-color: #FAF9F5;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    color: #3D3929;
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QComboBox {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    padding: 6px 10px;
    color: #141413;
    min-height: 28px;
}
QComboBox:hover {
    border-color: #C15F3C;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #FAF9F5;
    border: 1px solid #D5D3C8;
    selection-background-color: #E8D5C8;
    color: #141413;
}
QTabWidget::pane {
    border: 1px solid #D5D3C8;
    border-radius: 8px;
    background-color: #FAF9F5;
}
QTabBar::tab {
    background-color: #EEECE2;
    border: 1px solid #D5D3C8;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 6px 14px;
    color: #3D3929;
    font-weight: bold;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #FAF9F5;
    border-color: #C15F3C;
    color: #C15F3C;
}
QTabBar::tab:hover {
    background-color: #E4E2D8;
}
QLabel#statusLabel {
    font-size: 12px;
    padding: 2px 8px;
}
QLabel#balanceLabel {
    font-size: 16px;
    font-weight: bold;
    color: #C15F3C;
}
QFrame#separator {
    background-color: #D5D3C8;
    max-height: 1px;
}
"""

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1A1815;
    color: #E8E6E3;
    font-family: Georgia, Cambria, 'Times New Roman', serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #3A352B;
    border-radius: 12px;
    margin-top: 10px;
    padding-top: 16px;
    font-weight: bold;
    color: #B5AFA5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
}
QPushButton {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    border-radius: 8px;
    padding: 8px 20px;
    color: #E8E6E3;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #342E24;
    border-color: #C15F3C;
}
QPushButton:pressed {
    background-color: #3A352B;
}
QPushButton:disabled {
    background-color: #1A1815;
    color: #4A453B;
    border-color: #2A251D;
}
QPushButton#startBtn {
    background-color: #C15F3C;
    border: none;
    color: #FFFFFF;
}
QPushButton#startBtn:hover {
    background-color: #B5523C;
}
QPushButton#startBtn:disabled {
    background-color: #2A251D;
    color: #4A453B;
}
QPushButton#stopBtn {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    color: #E8E6E3;
}
QPushButton#stopBtn:hover {
    background-color: #342E24;
}
QPushButton#tradeBtn {
    background-color: #2A251D;
    border: 2px solid #C44444;
    color: #C44444;
}
QPushButton#tradeBtn:hover {
    background-color: #342E24;
}
QPushButton#emergencyBtn {
    background-color: #C44444;
    border: none;
    color: #FFFFFF;
}
QPushButton#emergencyBtn:hover {
    background-color: #A83838;
}
QLineEdit {
    background-color: #201D18;
    border: 1px solid #3A352B;
    border-radius: 8px;
    padding: 6px 10px;
    color: #E8E6E3;
}
QLineEdit:focus {
    border-color: #C15F3C;
}
QCheckBox {
    spacing: 6px;
    color: #E8E6E3;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4A453B;
    border-radius: 4px;
    background-color: #201D18;
}
QCheckBox::indicator:checked {
    background-color: #C15F3C;
    border-color: #B5523C;
}
QTableWidget {
    background-color: #201D18;
    border: 1px solid #3A352B;
    border-radius: 8px;
    gridline-color: #2A251D;
    selection-background-color: #3A352B;
    color: #E8E6E3;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #2A251D;
    color: #B5AFA5;
    border: none;
    border-bottom: 1px solid #3A352B;
    padding: 6px;
    font-weight: bold;
}
QTextEdit {
    background-color: #161411;
    border: 1px solid #3A352B;
    border-radius: 8px;
    color: #B5AFA5;
    font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
    font-size: 12px;
}
QComboBox {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    border-radius: 8px;
    padding: 6px 10px;
    color: #E8E6E3;
    min-height: 28px;
}
QComboBox:hover {
    border-color: #C15F3C;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #201D18;
    border: 1px solid #3A352B;
    selection-background-color: #3A352B;
    color: #E8E6E3;
}
QTabWidget::pane {
    border: 1px solid #3A352B;
    border-radius: 8px;
    background-color: #201D18;
}
QTabBar::tab {
    background-color: #2A251D;
    border: 1px solid #3A352B;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 6px 14px;
    color: #B5AFA5;
    font-weight: bold;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #201D18;
    border-color: #C15F3C;
    color: #C15F3C;
}
QTabBar::tab:hover {
    background-color: #342E24;
}
QLabel#statusLabel {
    font-size: 12px;
    padding: 2px 8px;
}
QLabel#balanceLabel {
    font-size: 16px;
    font-weight: bold;
    color: #C15F3C;
}
QFrame#separator {
    background-color: #3A352B;
    max-height: 1px;
}
"""


# ---------------------------------------------------------------------------
# Scan Engine (background thread)
# ---------------------------------------------------------------------------
class ScanEngine(QObject):
    """Background ZeroPoint scanner: fetches MT5 data, detects signals, places trades."""

    log_message = Signal(str)
    signal_detected = Signal(str, str, float, float, float, float, float, float)
    # symbol, direction, entry, sl, tp1, tp2, tp3, atr
    scan_complete = Signal(str, object)
    # symbol, enriched DataFrame

    MAX_SIGNAL_AGE = 6   # max bars old a signal can be to still enter

    def __init__(self):
        super().__init__()
        self._running = False
        self._auto_trade = False     # only place trades when True
        self._last_signal_dir = {}   # {symbol: 1 or -1}
        self._last_bar_time = {}     # {symbol: datetime} — dedup bar
        self._entered_signals = {}   # {symbol: bar_time} — signals we've already traded
        # V4 defaults (overridden by set_v4_params from UI)
        self._v4 = {
            "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
            "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
            "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
            "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
        }

    def set_v4_params(self, params: dict):
        """Update V4 profit capture parameters from UI settings."""
        self._v4.update(params)

    # ── Trade helpers (ported from webhook_bridge.py) ──

    def _resolve_symbol(self, ticker: str):
        candidates = [ticker, ticker + ".raw", ticker + "m",
                      ticker + ".a", ticker + ".e", ticker[:6]]
        for c in candidates:
            info = mt5.symbol_info(c)
            if info is not None:
                mt5.symbol_select(c, True)
                return c
        self.log_message.emit(f"Cannot resolve symbol: {ticker}")
        return None

    def _check_existing_position(self, symbol: str) -> bool:
        positions = mt5.positions_get()
        if not positions:
            return False
        norm = symbol.upper().replace(".", "").replace("#", "")
        for pos in positions:
            pos_norm = pos.symbol.upper().replace(".", "").replace("#", "")
            if pos_norm == norm:
                return True
        return False

    def _calc_lot_size(self, entry, sl, sym_info, risk_pct, default_lot):
        try:
            acct = mt5.account_info()
            if acct is None:
                return default_lot

            balance = acct.balance
            risk_amount = balance * risk_pct

            point = sym_info.point
            tick_size = sym_info.trade_tick_size or point
            tick_value = sym_info.trade_tick_value
            if tick_value <= 0:
                tick_value = sym_info.trade_contract_size * tick_size

            sl_distance = abs(entry - sl)
            sl_ticks = sl_distance / tick_size if tick_size > 0 else 0
            loss_per_lot = sl_ticks * tick_value

            if loss_per_lot <= 0:
                return default_lot

            lot = risk_amount / loss_per_lot
            vol_step = sym_info.volume_step
            lot = round(lot / vol_step) * vol_step
            lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))

            cap_table = [
                (500, 0.10), (1000, 0.20), (3000, 0.50),
                (5000, 1.00), (10000, 2.00), (50000, 5.00),
                (float('inf'), 10.00),
            ]
            for threshold, max_lot in cap_table:
                if balance <= threshold:
                    lot = min(lot, max_lot)
                    break

            self.log_message.emit(f"  Lot: balance=${balance:.0f} risk=${risk_amount:.0f} lot={lot:.2f}")
            return lot
        except Exception as e:
            self.log_message.emit(f"  Lot calc error: {e}, using {default_lot}")
            return default_lot

    def _place_trade(self, symbol_resolved, action, lot, sl, tp1, sym_info):
        try:
            if action == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = sym_info.ask
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = sym_info.bid

            digits = sym_info.digits
            sl_r = round(sl, digits)
            tp_r = round(tp1, digits)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym_info.name,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl_r,
                "tp": tp_r,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": f"ACi-ZP-{action}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }

            for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill_mode
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.log_message.emit(
                        f"  TRADE PLACED: {action} {sym_info.name} "
                        f"{lot:.2f}L @ {price:.5f} | SL={sl_r} TP={tp_r}"
                    )
                    return True

            rc = result.retcode if result else "?"
            self.log_message.emit(f"  Trade FAILED: {sym_info.name} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  Trade error: {e}")
            return False

    # ── Main scan loop ──

    def start_scanning(self, symbols, tf_key, poll_interval, risk_pct, default_lot, max_trades):
        self._symbols = symbols
        self._tf_key = tf_key
        self._tf_mt5 = TF_MAP.get(tf_key, mt5.TIMEFRAME_H4)
        self._poll = poll_interval
        self._risk_pct = risk_pct
        self._default_lot = default_lot
        self._max_trades = max_trades
        self._running = True

        t = threading.Thread(target=self._scan_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _scan_loop(self):
        self.log_message.emit(f"Scanner started | {self._tf_key} | {len(self._symbols)} symbols | poll={self._poll}s")
        first_scan = True

        while self._running:
            scanned = 0
            for symbol in self._symbols:
                if not self._running:
                    break
                try:
                    self._scan_symbol(symbol)
                    scanned += 1
                except Exception as e:
                    self.log_message.emit(f"[{symbol}] Scan error: {e}")

            if first_scan:
                self.log_message.emit(f"Scan complete: {scanned}/{len(self._symbols)} pairs checked")
                first_scan = False

            # Interruptible sleep
            for _ in range(self._poll):
                if not self._running:
                    break
                _time.sleep(1)

        self.log_message.emit("Scanner stopped.")

    def _scan_symbol(self, symbol):
        resolved = self._resolve_symbol(symbol)
        if resolved is None:
            return

        # Fetch OHLCV — 500 bars for full historical view
        rates = mt5.copy_rates_from_pos(resolved, self._tf_mt5, 0, 500)
        if rates is None or len(rates) < 20:
            return

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        # Compute ZeroPoint state
        df_zp = compute_zeropoint_state(df)
        if df_zp is None or len(df_zp) < 2:
            return

        # Emit data for chart (always, for every pair)
        self.scan_complete.emit(symbol, df_zp)

        # Current ZP direction from last bar
        last_pos = int(df_zp.iloc[-2]["pos"]) if len(df_zp) > 1 else 0
        pos_str = "BULL" if last_pos == 1 else "BEAR"

        # ── Find most recent signal within MAX_SIGNAL_AGE confirmed bars ──
        # iloc[-1] is the forming (current) bar, so confirmed bars are iloc[-2], iloc[-3], ...
        n = len(df_zp)
        signal_bar = None
        signal_age = 0   # how many bars ago (0 = last confirmed bar)

        for lookback in range(2, 2 + self.MAX_SIGNAL_AGE):
            idx = n - lookback
            if idx < 0:
                break
            row = df_zp.iloc[idx]
            if row.get("buy_signal", False) or row.get("sell_signal", False):
                signal_bar = row
                signal_age = lookback - 2  # 0 = most recent confirmed bar
                break

        if signal_bar is None:
            return

        bar_time = signal_bar["time"]

        # Already traded this exact signal?
        if self._entered_signals.get(symbol) == bar_time:
            return

        buy_sig = bool(signal_bar.get("buy_signal", False))
        sell_sig = bool(signal_bar.get("sell_signal", False))
        direction = "BUY" if buy_sig else "SELL"
        entry_price = float(signal_bar["close"])  # original signal price
        sl_val = float(signal_bar["smart_sl"]) if not np.isnan(signal_bar["smart_sl"]) else None
        atr_val = float(signal_bar["atr"]) if not np.isnan(signal_bar["atr"]) else None

        if sl_val is None or atr_val is None:
            return

        # V4 optimized TP levels (read from UI settings)
        _tp1m = self._v4["tp1_mult"]
        _tp2m = self._v4["tp2_mult"]
        _tp3m = self._v4["tp3_mult"]
        if direction == "BUY":
            tp1 = entry_price + atr_val * _tp1m
            tp2 = entry_price + atr_val * _tp2m
            tp3 = entry_price + atr_val * _tp3m
        else:
            tp1 = entry_price - atr_val * _tp1m
            tp2 = entry_price - atr_val * _tp2m
            tp3 = entry_price - atr_val * _tp3m

        # R:R check
        sl_dist = abs(entry_price - sl_val)
        tp_dist = abs(tp1 - entry_price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < 0.3:
            self.log_message.emit(f"[{symbol}] {direction} R:R too low ({rr:.2f}), skipping")
            return

        # Staleness check — current price vs signal price
        # If price has drifted more than 50% of SL distance, skip (too stale)
        sym_info = mt5.symbol_info(resolved)
        if sym_info is None:
            return
        current_price = sym_info.ask if direction == "BUY" else sym_info.bid
        drift = abs(current_price - entry_price)
        if sl_dist > 0 and drift > sl_dist * 0.50:
            if signal_age == 0:
                self.log_message.emit(f"[{symbol}] {direction} price drifted {drift:.5f} (>{sl_dist*0.50:.5f}), too stale")
            return

        # Check signal hasn't already been invalidated (price past SL)
        if direction == "BUY" and current_price < sl_val:
            return
        if direction == "SELL" and current_price > sl_val:
            return

        age_str = f" ({signal_age} bars ago)" if signal_age > 0 else ""
        self.log_message.emit(f"[{symbol}] ZP {direction}{age_str} | entry={entry_price:.5f} SL={sl_val:.5f} "
                              f"TP1={tp1:.5f} R:R={rr:.2f} ATR={atr_val:.5f}")

        # Always emit signal for chart drawing (even when trading is off)
        self.signal_detected.emit(symbol, direction, entry_price, sl_val, tp1, tp2, tp3, atr_val)

        # Only place trade if auto-trade is ON
        if not self._auto_trade:
            self.log_message.emit(f"[{symbol}] Trading OFF — signal detected but not executing")
            self._entered_signals[symbol] = bar_time  # mark so we don't spam logs
            return

        # Check existing position
        if self._check_existing_position(symbol):
            self.log_message.emit(f"[{symbol}] Already have position, skipping")
            self._entered_signals[symbol] = bar_time
            return

        # Check max concurrent
        positions = mt5.positions_get()
        if positions and len(positions) >= self._max_trades:
            self.log_message.emit(f"Max concurrent ({self._max_trades}) reached, skipping")
            return

        # Margin safety check
        acct = mt5.account_info()
        if acct and acct.margin_level > 0 and acct.margin_level < MIN_MARGIN_LEVEL_PCT:
            self.log_message.emit(f"[{symbol}] Margin level {acct.margin_level:.0f}% < {MIN_MARGIN_LEVEL_PCT}%, skipping")
            return

        # Check free margin would stay positive
        if acct and acct.margin_free is not None and acct.margin_free < acct.balance * 0.20:
            self.log_message.emit(f"[{symbol}] Free margin ${acct.margin_free:.0f} too low (<20% of balance), skipping")
            return

        # Place trade at CURRENT price (not signal price)
        lot = self._calc_lot_size(current_price, sl_val, sym_info, self._risk_pct, self._default_lot)
        if lot <= 0:
            return

        success = self._place_trade(resolved, direction, lot, sl_val, tp1, sym_info)
        if success:
            self._entered_signals[symbol] = bar_time  # mark as traded
            self.log_message.emit(f"[{symbol}] Trade executed: {direction} {lot:.2f}L{age_str}")


# ---------------------------------------------------------------------------
# V4 TP Manager — Full 5-layer profit capture system
# Layers: Micro-Partial | Early BE | Tighter TPs | Post-TP1 Trail | Stall Exit
# ---------------------------------------------------------------------------
class TPManager(QObject):
    """V4 Trade Manager: 5-layer profit capture matching backtested system."""

    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._trade_info = {}   # {ticket: {direction, entry, sl, atr, tp1-3, original_lot}}
        self._tp_state = {}     # {ticket: {tp1, tp2, tp3, micro_tp, be_activated, stall_be, ...}}
        # V4 defaults (overridden by set_v4_params from UI)
        self._v4 = {
            "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
            "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
            "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
            "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
        }

    def set_v4_params(self, params: dict):
        """Update V4 profit capture parameters from UI settings."""
        self._v4.update(params)

    def register_trade(self, ticket, direction, entry, sl, tp1, tp2, tp3, atr_val, original_lot):
        """Register a new trade for V4 profit capture management."""
        self._trade_info[ticket] = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "atr": atr_val,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "original_lot": original_lot,
        }
        self._tp_state[ticket] = {
            "tp1": False, "tp2": False, "tp3": False,
            "micro_tp": False,           # Micro-partial at micro_tp * ATR
            "be_activated": False,       # Early BE at be_trigger * ATR
            "stall_be": False,           # Stall exit after stall_bars checks
            "profit_trail_sl": None,     # Post-TP1 trailing SL
            "max_favorable": entry,      # Best price seen
            "checks": 0,                 # ~bars since entry (each check = 2s, H4 = 14400s)
        }
        v = self._v4
        self.log_message.emit(
            f"  V4 Manager: #{ticket} | TP1={tp1:.5f} TP2={tp2:.5f} TP3={tp3:.5f} | "
            f"BE@{v['be_trigger']}x Stall@{v['stall_bars']}bars Trail@{v['trail_dist']}x"
        )

    def start(self):
        self._running = True
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _monitor_loop(self):
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                self.log_message.emit(f"V4 Manager error: {e}")
            _time.sleep(2)

    def _check_positions(self):
        positions = mt5.positions_get()
        if not positions:
            return

        # Clean up closed trades
        open_tickets = {p.ticket for p in positions}
        closed = [t for t in self._trade_info if t not in open_tickets]
        for t in closed:
            self._trade_info.pop(t, None)
            self._tp_state.pop(t, None)

        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue
            ticket = pos.ticket
            if ticket not in self._trade_info:
                continue

            info = self._trade_info[ticket]
            state = self._tp_state[ticket]
            current = pos.price_current
            direction = info["direction"]
            entry = info["entry"]
            atr = info["atr"]
            original_lot = info["original_lot"]
            is_buy = direction == "BUY"

            sym_info = mt5.symbol_info(pos.symbol)
            if not sym_info:
                continue

            state["checks"] += 1

            # Track max favorable price
            if is_buy:
                if current > state["max_favorable"]:
                    state["max_favorable"] = current
                cur_profit_dist = current - entry
            else:
                if current < state["max_favorable"]:
                    state["max_favorable"] = current
                cur_profit_dist = entry - current

            max_profit_dist = abs(state["max_favorable"] - entry)

            # Local V4 params from UI settings
            _v = self._v4
            _micro_tp_mult = _v["micro_tp"]
            _micro_pct = _v["micro_pct"]
            _be_trigger = _v["be_trigger"]
            _be_buffer = _v["be_buffer"]
            _stall_bars = _v["stall_bars"]
            _trail_dist_mult = _v["trail_dist"]

            # ── Layer 1: Micro-Partial at micro_tp * ATR ──
            if not state["micro_tp"] and not state["tp1"]:
                micro_level = entry + (_micro_tp_mult * atr if is_buy else -_micro_tp_mult * atr)
                micro_hit = (is_buy and current >= micro_level) or (not is_buy and current <= micro_level)
                if micro_hit:
                    close_lot = round(original_lot * _micro_pct, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    close_lot = min(close_lot, pos.volume)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["micro_tp"] = True
                            self.log_message.emit(
                                f"  MICRO-TP: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} ({_micro_tp_mult}x ATR)")

            # ── Layer 2: Early BE at be_trigger * ATR ──
            if not state["be_activated"]:
                if max_profit_dist >= _be_trigger * atr:
                    be_level = entry + (_be_buffer * atr if is_buy else -_be_buffer * atr)
                    # Only move SL if it improves (tighter)
                    should_move = (is_buy and be_level > pos.sl) or (not is_buy and be_level < pos.sl)
                    if should_move:
                        ok = self._move_sl(pos, be_level, sym_info)
                        if ok:
                            state["be_activated"] = True
                            self.log_message.emit(
                                f"  EARLY BE: {pos.symbol} SL -> {be_level:.5f} "
                                f"(entry+{_be_buffer}x ATR)")

            # ── Layer 3: Stall Exit — move to BE after stall_bars H4 bars w/o TP1 ──
            if not state["tp1"] and not state["stall_be"]:
                # Approximate bars: each check is 2s, H4 bar = 14400s
                approx_bars = state["checks"] * 2 / 14400
                if approx_bars >= _stall_bars:
                    be_level = entry + (_be_buffer * atr if is_buy else -_be_buffer * atr)
                    should_move = (is_buy and be_level > pos.sl) or (not is_buy and be_level < pos.sl)
                    if should_move:
                        ok = self._move_sl(pos, be_level, sym_info)
                        if ok:
                            state["stall_be"] = True
                            state["be_activated"] = True
                            self.log_message.emit(
                                f"  STALL EXIT: {pos.symbol} SL -> BE after ~{approx_bars:.1f} bars w/o TP1")

            # ── Layer 4: TP1 Hit — partial close + activate profit trail ──
            if not state["tp1"]:
                hit = (is_buy and current >= info["tp1"]) or (not is_buy and current <= info["tp1"])
                if hit:
                    close_lot = round(original_lot * TP1_CLOSE_PCT, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    # Refresh volume (micro-partial may have reduced it)
                    refreshed = mt5.positions_get(ticket=ticket)
                    cur_vol = refreshed[0].volume if refreshed and len(refreshed) > 0 else pos.volume
                    close_lot = min(close_lot, cur_vol)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["tp1"] = True
                            # Move SL to BE if not already
                            if not state["be_activated"]:
                                be_level = entry + (_be_buffer * atr if is_buy else -_be_buffer * atr)
                                self._move_sl(pos, be_level, sym_info)
                                state["be_activated"] = True
                            self.log_message.emit(
                                f"  TP1 HIT: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} | Trail activated")

            # ── Layer 5: Post-TP1 Profit Trail ──
            if state["tp1"] and not state["tp3"]:
                trail_dist = _trail_dist_mult * atr
                if is_buy:
                    new_trail = state["max_favorable"] - trail_dist
                    if new_trail > entry and (state["profit_trail_sl"] is None or new_trail > state["profit_trail_sl"]):
                        state["profit_trail_sl"] = new_trail
                        # Move SL to trail level
                        if new_trail > pos.sl:
                            ok = self._move_sl(pos, new_trail, sym_info)
                            if ok:
                                self.log_message.emit(
                                    f"  TRAIL: {pos.symbol} SL -> {new_trail:.5f} "
                                    f"(peak={state['max_favorable']:.5f} - {_trail_dist_mult}x ATR)")
                else:
                    new_trail = state["max_favorable"] + trail_dist
                    if new_trail < entry and (state["profit_trail_sl"] is None or new_trail < state["profit_trail_sl"]):
                        state["profit_trail_sl"] = new_trail
                        if new_trail < pos.sl:
                            ok = self._move_sl(pos, new_trail, sym_info)
                            if ok:
                                self.log_message.emit(
                                    f"  TRAIL: {pos.symbol} SL -> {new_trail:.5f} "
                                    f"(peak={state['max_favorable']:.5f} + {_trail_dist_mult}x ATR)")

            # ── TP2: partial close + tighten trail ──
            if state["tp1"] and not state["tp2"]:
                hit = (is_buy and current >= info["tp2"]) or (not is_buy and current <= info["tp2"])
                if hit:
                    refreshed = mt5.positions_get(ticket=ticket)
                    if refreshed and len(refreshed) > 0:
                        cur_vol = refreshed[0].volume
                    else:
                        continue
                    close_lot = round(original_lot * TP2_CLOSE_PCT, 2)
                    vol_step = sym_info.volume_step
                    vol_min = sym_info.volume_min
                    close_lot = max(vol_min, round(close_lot / vol_step) * vol_step)
                    close_lot = min(close_lot, cur_vol)
                    if close_lot >= vol_min:
                        ok = self._partial_close(pos, close_lot, sym_info)
                        if ok:
                            state["tp2"] = True
                            # Move SL to TP1 level
                            self._move_sl(pos, info["tp1"], sym_info)
                            self.log_message.emit(
                                f"  TP2 HIT: {pos.symbol} closed {close_lot:.2f}L "
                                f"@ {current:.5f} | SL -> TP1")

            # ── TP3: close everything ──
            if state["tp2"] and not state["tp3"]:
                hit = (is_buy and current >= info["tp3"]) or (not is_buy and current <= info["tp3"])
                if hit:
                    refreshed = mt5.positions_get(ticket=ticket)
                    if refreshed and len(refreshed) > 0:
                        cur_vol = refreshed[0].volume
                    else:
                        continue
                    if cur_vol > 0:
                        ok = self._partial_close(pos, cur_vol, sym_info)
                        if ok:
                            state["tp3"] = True
                            self.log_message.emit(
                                f"  TP3 HIT: {pos.symbol} CLOSED ALL {cur_vol:.2f}L "
                                f"@ {current:.5f}")

    def _partial_close(self, pos, close_volume, sym_info):
        """Partially close a position."""
        try:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            price = sym_info.bid if pos.type == 0 else sym_info.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": MAGIC_NUMBER,
                "comment": "ACi-TP-Partial",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            for fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                request["type_filling"] = fill
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    return True
            rc = result.retcode if result else "?"
            self.log_message.emit(f"  Partial close FAILED: {pos.symbol} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  Partial close error: {e}")
            return False

    def _move_sl_to_be(self, pos, entry, sym_info):
        """Move SL to breakeven (entry price)."""
        self._move_sl(pos, entry, sym_info)

    def _move_sl(self, pos, new_sl, sym_info):
        """Modify position SL."""
        try:
            digits = sym_info.digits
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "sl": round(new_sl, digits),
                "tp": pos.tp,
                "magic": MAGIC_NUMBER,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            rc = result.retcode if result else "?"
            self.log_message.emit(f"  SL modify FAILED: {pos.symbol} retcode={rc}")
            return False
        except Exception as e:
            self.log_message.emit(f"  SL modify error: {e}")
            return False


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
class ACiApp(QMainWindow):
    _log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACi  —  V4 ZeroPoint Pro  |  97.5% WR")
        self.setMinimumSize(900, 700)
        self.resize(1200, 850)

        self._dark_mode = True
        self._scanner = None
        self._running = False
        self._trading_enabled = False
        self._chart_data = {}       # {symbol: DataFrame}  — cached for chart
        self._chart_initialized = {}  # {symbol: bool}
        self._starting_balance = None  # set on first MT5 connect for growth calc
        self._positions_dialog = None

        # TP Manager
        self._tp_manager = TPManager()
        self._tp_manager.log_message.connect(self._on_scanner_log)
        self._tp_manager.start()

        self._log_signal.connect(self._log_on_main_thread)

        self._build_ui()
        self._load_settings()
        # Push V4 params to TP manager after UI is built + settings loaded
        self._tp_manager.set_v4_params(self._get_v4_params())
        self.setStyleSheet(DARK_STYLE)

        # MT5 check on startup
        self._check_mt5()

        # Live position refresh
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_live)
        self._timer.start(2000)

    # ── Build UI ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(8)

        # --- Status bar row 1: connection + scanner ---
        status_row = QHBoxLayout()
        self.lbl_mt5 = QLabel("MT5: --")
        self.lbl_mt5.setObjectName("statusLabel")
        status_row.addWidget(self.lbl_mt5)

        self.lbl_scanner = QLabel("Scanner: OFF")
        self.lbl_scanner.setObjectName("statusLabel")
        status_row.addWidget(self.lbl_scanner)

        status_row.addStretch()

        # Account stats row
        self.lbl_balance = QLabel("Bal: $0.00")
        self.lbl_balance.setObjectName("balanceLabel")
        status_row.addWidget(self.lbl_balance)

        self.lbl_equity = QLabel("Eq: $0.00")
        self.lbl_equity.setObjectName("statusLabel")
        self.lbl_equity.setStyleSheet("font-weight: bold; font-size: 13px;")
        status_row.addWidget(self.lbl_equity)

        self.lbl_margin = QLabel("Margin: $0.00")
        self.lbl_margin.setObjectName("statusLabel")
        status_row.addWidget(self.lbl_margin)

        self.lbl_free_margin = QLabel("Free: $0.00")
        self.lbl_free_margin.setObjectName("statusLabel")
        status_row.addWidget(self.lbl_free_margin)

        self.lbl_growth = QLabel("Growth: 0.0%")
        self.lbl_growth.setObjectName("statusLabel")
        self.lbl_growth.setStyleSheet("font-weight: bold; font-size: 13px;")
        status_row.addWidget(self.lbl_growth)

        main_layout.addLayout(status_row)

        # --- Separator ---
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(sep)

        # --- Button row ---
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("Start Scanner")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_stop)

        self.btn_trade = QPushButton("Trade: OFF")
        self.btn_trade.setObjectName("tradeBtn")
        self.btn_trade.clicked.connect(self._toggle_trading)
        btn_row.addWidget(self.btn_trade)

        self.btn_positions = QPushButton("Positions")
        self.btn_positions.clicked.connect(self._show_positions_dialog)
        btn_row.addWidget(self.btn_positions)

        self.btn_theme = QPushButton("Light Mode")
        self.btn_theme.clicked.connect(self._toggle_theme)
        btn_row.addWidget(self.btn_theme)

        btn_row.addStretch()

        self.btn_close_all = QPushButton("Close All")
        self.btn_close_all.setObjectName("emergencyBtn")
        self.btn_close_all.clicked.connect(self._on_close_all)
        btn_row.addWidget(self.btn_close_all)

        main_layout.addLayout(btn_row)

        # --- Trade Execution Settings (3 rows in one group) ---
        settings_group = QGroupBox("Trade Execution  —  V4 Profit Capture")
        settings_main = QVBoxLayout(settings_group)
        settings_main.setSpacing(6)

        # ── Row 1: Core execution params ──
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        row1.addWidget(QLabel("Risk %:"))
        self.inp_risk = QLineEdit("30")
        self.inp_risk.setFixedWidth(40)
        self.inp_risk.setToolTip("Risk per trade (30% = half-Kelly for 97.5% WR)")
        row1.addWidget(self.inp_risk)

        row1.addWidget(QLabel("Lots:"))
        self.inp_lots = QLineEdit("0.40")
        self.inp_lots.setFixedWidth(50)
        self.inp_lots.setToolTip("Default lot size (overridden by risk-based sizing)")
        row1.addWidget(self.inp_lots)

        row1.addWidget(QLabel("Max:"))
        self.inp_max_trades = QLineEdit("5")
        self.inp_max_trades.setFixedWidth(30)
        self.inp_max_trades.setToolTip("Max concurrent trades across all pairs")
        row1.addWidget(self.inp_max_trades)

        row1.addWidget(QLabel("Poll (sec):"))
        self.inp_poll = QLineEdit("30")
        self.inp_poll.setFixedWidth(40)
        row1.addWidget(self.inp_poll)

        row1.addWidget(QLabel("Timeframe:"))
        self.combo_tf = QComboBox()
        self.combo_tf.setFixedWidth(70)
        for tf in TIMEFRAMES:
            self.combo_tf.addItem(tf, tf)
        self.combo_tf.setCurrentIndex(5)  # H4 default
        self.combo_tf.currentIndexChanged.connect(self._on_tf_changed)
        row1.addWidget(self.combo_tf)

        row1.addStretch()
        settings_main.addLayout(row1)

        # ── Thin separator ──
        sep_h = QFrame()
        sep_h.setFrameShape(QFrame.Shape.HLine)
        sep_h.setStyleSheet("color: #3A352B;")
        settings_main.addWidget(sep_h)

        # ── Row 2: V4 TPs + Breakeven ──
        _tp = "color: #4ADE80; font-weight: bold; font-size: 12px;"
        _be = "color: #FBBF24; font-weight: bold; font-size: 12px;"
        _mi = "color: #38BDF8; font-weight: bold; font-size: 12px;"
        _tr = "color: #A78BFA; font-weight: bold; font-size: 12px;"
        _u = "font-size: 11px;"

        row2 = QHBoxLayout()
        row2.setSpacing(12)

        lbl = QLabel("TP1:"); lbl.setStyleSheet(_tp); row2.addWidget(lbl)
        self.inp_tp1 = QLineEdit(str(TP1_MULT_AGG))
        self.inp_tp1.setFixedWidth(40)
        self.inp_tp1.setToolTip("TP1 distance in ATR multiples (V4: 0.8)")
        row2.addWidget(self.inp_tp1)
        lbl = QLabel("x"); lbl.setStyleSheet(_u); row2.addWidget(lbl)

        lbl = QLabel("TP2:"); lbl.setStyleSheet(_tp); row2.addWidget(lbl)
        self.inp_tp2 = QLineEdit(str(TP2_MULT_AGG))
        self.inp_tp2.setFixedWidth(40)
        self.inp_tp2.setToolTip("TP2 distance in ATR multiples (V4: 2.0)")
        row2.addWidget(self.inp_tp2)
        lbl = QLabel("x"); lbl.setStyleSheet(_u); row2.addWidget(lbl)

        lbl = QLabel("TP3:"); lbl.setStyleSheet(_tp); row2.addWidget(lbl)
        self.inp_tp3 = QLineEdit(str(TP3_MULT_AGG))
        self.inp_tp3.setFixedWidth(40)
        self.inp_tp3.setToolTip("TP3 distance in ATR multiples (V4: 5.0)")
        row2.addWidget(self.inp_tp3)
        lbl = QLabel("x ATR"); lbl.setStyleSheet(_u); row2.addWidget(lbl)

        sep_v = QFrame(); sep_v.setFrameShape(QFrame.Shape.VLine)
        sep_v.setStyleSheet("color: #3A352B;"); row2.addWidget(sep_v)

        lbl = QLabel("BE:"); lbl.setStyleSheet(_be); row2.addWidget(lbl)
        self.inp_be_trigger = QLineEdit(str(BE_TRIGGER_MULT))
        self.inp_be_trigger.setFixedWidth(36)
        self.inp_be_trigger.setToolTip("Move SL to breakeven at this ATR multiple (0.5)")
        row2.addWidget(self.inp_be_trigger)
        lbl = QLabel("x"); lbl.setStyleSheet(_u); row2.addWidget(lbl)

        lbl = QLabel("Buf:"); lbl.setStyleSheet(_be); row2.addWidget(lbl)
        self.inp_be_buffer = QLineEdit(str(BE_BUFFER_MULT))
        self.inp_be_buffer.setFixedWidth(36)
        self.inp_be_buffer.setToolTip("BE SL set to entry + buffer (0.15 = slight profit)")
        row2.addWidget(self.inp_be_buffer)
        lbl = QLabel("x ATR"); lbl.setStyleSheet(_u); row2.addWidget(lbl)

        row2.addStretch()
        settings_main.addLayout(row2)

        # ── Row 3: Micro-partial + Stall + Trail + live status ──
        row3 = QHBoxLayout()
        row3.setSpacing(12)

        lbl = QLabel("Micro:"); lbl.setStyleSheet(_mi); row3.addWidget(lbl)
        self.inp_micro_tp = QLineEdit(str(MICRO_TP_MULT))
        self.inp_micro_tp.setFixedWidth(36)
        self.inp_micro_tp.setToolTip("Take micro-partial at this ATR distance (0.8x)")
        row3.addWidget(self.inp_micro_tp)
        lbl = QLabel("x ATR"); lbl.setStyleSheet(_u); row3.addWidget(lbl)

        lbl = QLabel("Take:"); lbl.setStyleSheet(_mi); row3.addWidget(lbl)
        self.inp_micro_pct = QLineEdit(str(int(MICRO_TP_PCT * 100)))
        self.inp_micro_pct.setFixedWidth(30)
        self.inp_micro_pct.setToolTip("Percent of lot to close at micro-TP (15%)")
        row3.addWidget(self.inp_micro_pct)
        lbl = QLabel("%"); lbl.setStyleSheet(_u); row3.addWidget(lbl)

        sep_v2 = QFrame(); sep_v2.setFrameShape(QFrame.Shape.VLine)
        sep_v2.setStyleSheet("color: #3A352B;"); row3.addWidget(sep_v2)

        lbl = QLabel("Stall:"); lbl.setStyleSheet(_tr); row3.addWidget(lbl)
        self.inp_stall = QLineEdit(str(STALL_BARS))
        self.inp_stall.setFixedWidth(30)
        self.inp_stall.setToolTip("Move SL to BE after this many H4 bars without TP1 (6)")
        row3.addWidget(self.inp_stall)
        lbl = QLabel("bars"); lbl.setStyleSheet(_u); row3.addWidget(lbl)

        lbl = QLabel("Trail:"); lbl.setStyleSheet(_tr); row3.addWidget(lbl)
        self.inp_trail = QLineEdit(str(PROFIT_TRAIL_DISTANCE_MULT))
        self.inp_trail.setFixedWidth(36)
        self.inp_trail.setToolTip("Post-TP1 trailing SL distance in ATR multiples (0.8)")
        row3.addWidget(self.inp_trail)
        lbl = QLabel("x ATR"); lbl.setStyleSheet(_u); row3.addWidget(lbl)

        sep_v3 = QFrame(); sep_v3.setFrameShape(QFrame.Shape.VLine)
        sep_v3.setStyleSheet("color: #3A352B;"); row3.addWidget(sep_v3)

        self.lbl_v4_active = QLabel("V4: Waiting...")
        self.lbl_v4_active.setStyleSheet("font-weight: bold; font-size: 12px; color: #6B6355;")
        row3.addWidget(self.lbl_v4_active)

        row3.addStretch()
        settings_main.addLayout(row3)

        # Live-update V4 params when any field changes
        for inp in (self.inp_tp1, self.inp_tp2, self.inp_tp3,
                    self.inp_be_trigger, self.inp_be_buffer,
                    self.inp_micro_tp, self.inp_micro_pct,
                    self.inp_stall, self.inp_trail):
            inp.editingFinished.connect(self._push_v4_params)

        main_layout.addWidget(settings_group)

        # --- Pairs ---
        pairs_group = QGroupBox("Trading Pairs")
        pairs_layout = QHBoxLayout(pairs_group)
        self.pair_checks = {}
        for sym in ALL_PAIRS:
            cb = QCheckBox(sym)
            cb.setChecked(True)
            self.pair_checks[sym] = cb
            pairs_layout.addWidget(cb)
        pairs_layout.addStretch()
        main_layout.addWidget(pairs_group)

        # --- Charts (QTabWidget with lightweight-charts) ---
        self.chart_tabs = QTabWidget()
        self.chart_tabs.setMinimumHeight(350)
        self.chart_objects = {}       # {symbol: QtChart}
        self.chart_lines = {}         # {symbol: trailing_stop Line}
        self.chart_sl_lines = {}      # {symbol: list of HorizontalLine}
        self._chart_loaded = {}       # {symbol: bool}

        for sym in ALL_PAIRS:
            chart = QtChart(self)
            chart.legend(visible=True, font_size=12)
            chart.layout(
                background_color="#1A1815",
                text_color="#B5AFA5",
                font_size=12,
                font_family="Georgia",
            )
            chart.candle_style(
                up_color="#4A8C5D", down_color="#C15F3C",
                wick_up_color="#4A8C5D", wick_down_color="#C15F3C",
            )
            chart.volume_config(up_color="rgba(74,140,93,0.3)", down_color="rgba(193,95,60,0.3)")
            chart.crosshair(mode="normal")
            chart.grid(vert_enabled=True, horz_enabled=True)
            chart.time_scale(right_offset=5)

            # ZP trailing stop lines — two colors for bull/bear
            zp_bull = chart.create_line(name="ZP Bull", color="#4ADE80", width=2, price_label=True)
            zp_bear = chart.create_line(name="ZP Bear", color="#F87171", width=2, price_label=True)
            self.chart_lines[sym] = (zp_bull, zp_bear)
            self.chart_objects[sym] = chart
            self._chart_loaded[sym] = False

            self.chart_tabs.addTab(chart.get_webview(), sym)

        main_layout.addWidget(self.chart_tabs, stretch=5)

        # --- Compact log at bottom ---
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(90)
        self.txt_log.setPlaceholderText("Trade log...")
        main_layout.addWidget(self.txt_log)

        # --- Hidden positions table (lives in popup dialog) ---
        self.tbl_positions = None  # created in dialog

    # ── Timeframe change ──

    def _on_tf_changed(self, _index):
        """Reset charts and restart scanner when timeframe changes."""
        tf_key = self.combo_tf.currentData()
        self._log(f"Timeframe changed to {tf_key}")

        # Reset all chart loaded flags so they fully reload
        for sym in ALL_PAIRS:
            self._chart_loaded[sym] = False

        # Clear cached chart data
        self._chart_data.clear()

        # If scanner is running, restart with new TF
        if self._running and self._scanner:
            self._scanner.stop()
            self._scanner = None

            symbols = [s for s, cb in self.pair_checks.items() if cb.isChecked()]
            poll = int(self.inp_poll.text() or "30")
            risk = float(self.inp_risk.text() or "30") / 100
            lots = float(self.inp_lots.text() or "0.40")
            max_trades = int(self.inp_max_trades.text() or "5")

            self._scanner = ScanEngine()
            self._scanner._auto_trade = self._trading_enabled
            self._scanner.log_message.connect(self._on_scanner_log)
            self._scanner.signal_detected.connect(self._on_signal)
            self._scanner.scan_complete.connect(self._on_scan_data)
            self._push_v4_params()
            self._scanner.start_scanning(symbols, tf_key, poll, risk, lots, max_trades)

    # ── MT5 check ──

    def _check_mt5(self):
        try:
            if not mt5.initialize():
                self._log("MT5 not connected. Please start MetaTrader 5.")
                self.lbl_mt5.setText("MT5: OFFLINE")
                return False
            acct = mt5.account_info()
            if acct is None:
                self._log("MT5: Cannot read account info")
                return False
            self.lbl_mt5.setText(f"MT5: {acct.login}")
            self.lbl_balance.setText(f"Bal: ${acct.balance:.2f}")
            self._starting_balance = acct.balance
            self._log(f"MT5 connected: Account {acct.login} | ${acct.balance:.2f}")
            return True
        except Exception as e:
            self._log(f"MT5 error: {e}")
            self.lbl_mt5.setText("MT5: ERROR")
            return False

    # ── V4 Params ──

    def _get_v4_params(self):
        """Read V4 profit capture parameters from UI input fields."""
        try:
            return {
                "tp1_mult": float(self.inp_tp1.text() or TP1_MULT_AGG),
                "tp2_mult": float(self.inp_tp2.text() or TP2_MULT_AGG),
                "tp3_mult": float(self.inp_tp3.text() or TP3_MULT_AGG),
                "be_trigger": float(self.inp_be_trigger.text() or BE_TRIGGER_MULT),
                "be_buffer": float(self.inp_be_buffer.text() or BE_BUFFER_MULT),
                "micro_tp": float(self.inp_micro_tp.text() or MICRO_TP_MULT),
                "micro_pct": float(self.inp_micro_pct.text() or str(MICRO_TP_PCT * 100)) / 100,
                "stall_bars": int(self.inp_stall.text() or STALL_BARS),
                "trail_dist": float(self.inp_trail.text() or PROFIT_TRAIL_DISTANCE_MULT),
            }
        except (ValueError, AttributeError):
            # Fallback to module defaults if UI parse fails
            return {
                "tp1_mult": TP1_MULT_AGG, "tp2_mult": TP2_MULT_AGG, "tp3_mult": TP3_MULT_AGG,
                "be_trigger": BE_TRIGGER_MULT, "be_buffer": BE_BUFFER_MULT,
                "micro_tp": MICRO_TP_MULT, "micro_pct": MICRO_TP_PCT,
                "stall_bars": STALL_BARS, "trail_dist": PROFIT_TRAIL_DISTANCE_MULT,
            }

    def _push_v4_params(self):
        """Push current V4 params from UI to scanner and TPManager."""
        v4 = self._get_v4_params()
        if self._scanner:
            self._scanner.set_v4_params(v4)
        self._tp_manager.set_v4_params(v4)

    # ── Start/Stop ──

    def _on_start(self):
        if self._running:
            return

        if not mt5.initialize():
            self._log("Cannot start — MT5 not connected")
            return

        symbols = [s for s, cb in self.pair_checks.items() if cb.isChecked()]
        if not symbols:
            self._log("No pairs selected!")
            return

        tf_key = self.combo_tf.currentData()
        poll = int(self.inp_poll.text() or "30")
        risk = float(self.inp_risk.text() or "30") / 100
        lots = float(self.inp_lots.text() or "0.40")
        max_trades = int(self.inp_max_trades.text() or "5")

        self._scanner = ScanEngine()
        self._scanner._auto_trade = self._trading_enabled
        self._scanner.log_message.connect(self._on_scanner_log)
        self._scanner.signal_detected.connect(self._on_signal)
        self._scanner.scan_complete.connect(self._on_scan_data)
        # Push V4 params from UI to scanner + TP manager
        self._push_v4_params()
        self._scanner.start_scanning(symbols, tf_key, poll, risk, lots, max_trades)

        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_scanner.setText("Scanner: RUNNING")
        self._save_settings()

    def _on_stop(self):
        if self._scanner:
            self._scanner.stop()
            self._scanner = None
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_scanner.setText("Scanner: OFF")

    # ── Trade toggle ──

    def _toggle_trading(self):
        self._trading_enabled = not self._trading_enabled
        if self._trading_enabled:
            self.btn_trade.setText("Trade: ON")
            self.btn_trade.setStyleSheet(
                "background-color: #4A8C5D; border: none; color: #FFFFFF; "
                "font-weight: bold; border-radius: 8px; padding: 8px 20px; min-height: 28px;")
            self._log("TRADING ENABLED — signals will auto-execute")
        else:
            self.btn_trade.setText("Trade: OFF")
            self.btn_trade.setStyleSheet("")  # reset to theme default
            self._log("TRADING DISABLED — scan-only mode")

        # Update scanner's auto-trade flag
        if self._scanner:
            self._scanner._auto_trade = self._trading_enabled
            if self._trading_enabled:
                # Clear entered signals so it picks up recent signals on next scan
                self._scanner._entered_signals.clear()

    # ── Scanner signal handlers ──

    def _on_scanner_log(self, msg):
        self._log(msg)

    def _on_signal(self, symbol, direction, entry, sl, tp1, tp2, tp3, atr):
        sl_dist = abs(entry - sl)
        rr = abs(tp1 - entry) / sl_dist if sl_dist > 0 else 0
        v4 = self._get_v4_params()
        be_price = entry + (v4["be_trigger"] * atr if direction == "BUY" else -v4["be_trigger"] * atr)
        self._log(f"*** V4 SIGNAL: {direction} {symbol} @ {entry:.5f} | "
                  f"SL={sl:.5f} TP1={tp1:.5f} TP2={tp2:.5f} TP3={tp3:.5f} | "
                  f"R:R={rr:.2f} BE@{be_price:.5f} ATR={atr:.5f} ***")

        # Register with V4 TP Manager if trading is enabled
        # Find the matching open position ticket (delay 3s for MT5 to register)
        if self._trading_enabled:
            QTimer.singleShot(3000, lambda: self._register_trade_for_tp(
                symbol, direction, entry, sl, tp1, tp2, tp3, atr))

        # Draw V4 levels on chart (Entry, SL, BE trigger, Micro-TP, TP1, TP2, TP3)
        if symbol in self.chart_objects:
            chart = self.chart_objects[symbol]
            try:
                # Clear old level lines
                for old_line in self.chart_sl_lines.get(symbol, []):
                    try:
                        old_line.delete()
                    except Exception:
                        pass

                # Calculate V4 levels from UI settings
                sign = 1 if direction == "BUY" else -1
                micro_level = entry + sign * v4["micro_tp"] * atr
                be_level = entry + sign * v4["be_trigger"] * atr

                lines = []
                lines.append(chart.horizontal_line(entry, color="#FFFFFF", width=1, style="dashed",
                             text=f"ENTRY {entry:.5f}"))
                lines.append(chart.horizontal_line(sl, color="#F87171", width=2, style="solid",
                             text=f"SL {sl:.5f}"))
                lines.append(chart.horizontal_line(be_level, color="#FBBF24", width=1, style="dashed",
                             text=f"BE Trigger {be_level:.5f}"))
                lines.append(chart.horizontal_line(tp1, color="#4ADE80", width=1, style="dashed",
                             text=f"TP1 {tp1:.5f} ({v4['tp1_mult']}x ATR)"))
                lines.append(chart.horizontal_line(tp2, color="#38BDF8", width=1, style="dashed",
                             text=f"TP2 {tp2:.5f} ({v4['tp2_mult']}x ATR)"))
                lines.append(chart.horizontal_line(tp3, color="#A78BFA", width=1, style="dashed",
                             text=f"TP3 {tp3:.5f} ({v4['tp3_mult']}x ATR)"))
                self.chart_sl_lines[symbol] = lines
            except Exception as e:
                self._log(f"[{symbol}] Level draw error: {e}")

    def _register_trade_for_tp(self, symbol, direction, entry, sl, tp1, tp2, tp3, atr_val):
        """Find the just-placed trade ticket and register it with V4 TP manager."""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            norm = symbol.upper().replace(".", "").replace("#", "")
            for pos in positions:
                pos_norm = pos.symbol.upper().replace(".", "").replace("#", "")
                if pos_norm == norm and pos.magic == MAGIC_NUMBER:
                    # Check it's not already tracked
                    if pos.ticket not in self._tp_manager._trade_info:
                        self._tp_manager.register_trade(
                            pos.ticket, direction, entry, sl, tp1, tp2, tp3, atr_val, pos.volume)
                        return
        except Exception as e:
            self._log(f"TP registration error: {e}")

    # ── Positions Dialog ──

    def _show_positions_dialog(self):
        """Open a popup window showing open positions + trade history."""
        if self._positions_dialog and self._positions_dialog.isVisible():
            self._positions_dialog.raise_()
            self._positions_dialog.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("ACi V4 — Positions & Trade Management")
        dlg.resize(900, 600)
        dlg.setStyleSheet(DARK_STYLE if self._dark_mode else CLAUDE_STYLE)
        layout = QVBoxLayout(dlg)

        # --- Account summary at top ---
        acct_row = QHBoxLayout()
        self._dlg_lbl_acct = QLabel("Account: --")
        self._dlg_lbl_acct.setStyleSheet("font-weight: bold; font-size: 14px;")
        acct_row.addWidget(self._dlg_lbl_acct)
        acct_row.addStretch()
        self._dlg_lbl_pnl = QLabel("Open P/L: $0.00")
        self._dlg_lbl_pnl.setStyleSheet("font-weight: bold; font-size: 14px;")
        acct_row.addWidget(self._dlg_lbl_pnl)
        layout.addLayout(acct_row)

        # --- Open positions table ---
        layout.addWidget(QLabel("Open Positions  —  V4 Profit Capture"))
        self._dlg_tbl_open = QTableWidget(0, 11)
        self._dlg_tbl_open.setHorizontalHeaderLabels(
            ["Symbol", "Dir", "Lots", "Entry", "Current", "P/L", "SL", "TP",
             "V4 Layers", "Trail SL", "Peak"])
        self._dlg_tbl_open.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._dlg_tbl_open.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dlg_tbl_open.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self._dlg_tbl_open)

        # --- Closed history table ---
        layout.addWidget(QLabel("Recent Trade History (today)"))
        self._dlg_tbl_history = QTableWidget(0, 8)
        self._dlg_tbl_history.setHorizontalHeaderLabels(
            ["Symbol", "Dir", "Lots", "Entry", "Close", "P/L", "Commission", "Time"])
        self._dlg_tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._dlg_tbl_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dlg_tbl_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self._dlg_tbl_history)

        self._positions_dialog = dlg

        # Populate immediately
        self._refresh_positions_dialog()

        # Refresh timer for the dialog
        self._dlg_timer = QTimer(dlg)
        self._dlg_timer.timeout.connect(self._refresh_positions_dialog)
        self._dlg_timer.start(2000)

        dlg.show()

    def _refresh_positions_dialog(self):
        """Refresh the positions dialog with live data."""
        if not self._positions_dialog or not self._positions_dialog.isVisible():
            return
        try:
            acct = mt5.account_info()
            if acct:
                self._dlg_lbl_acct.setText(
                    f"Account: {acct.login}  |  Bal: ${acct.balance:.2f}  |  "
                    f"Eq: ${acct.equity:.2f}  |  Margin: ${acct.margin:.2f}  |  "
                    f"Free: ${acct.margin_free:.2f}")

            # --- Open positions ---
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            my_pos = [p for p in positions if p.magic == MAGIC_NUMBER]

            total_pnl = sum(p.profit for p in my_pos)
            pnl_color = "#4ADE80" if total_pnl >= 0 else "#F87171"
            self._dlg_lbl_pnl.setText(f"Open P/L: ${total_pnl:.2f}")
            self._dlg_lbl_pnl.setStyleSheet(
                f"font-weight: bold; font-size: 14px; color: {pnl_color};")

            self._dlg_tbl_open.setRowCount(len(my_pos))
            for row, pos in enumerate(my_pos):
                direction = "BUY" if pos.type == 0 else "SELL"
                pnl = pos.profit
                self._dlg_tbl_open.setItem(row, 0, QTableWidgetItem(pos.symbol))
                dir_item = QTableWidgetItem(direction)
                dir_item.setForeground(QColor("#4ADE80") if direction == "BUY" else QColor("#F87171"))
                self._dlg_tbl_open.setItem(row, 1, dir_item)
                self._dlg_tbl_open.setItem(row, 2, QTableWidgetItem(f"{pos.volume:.2f}"))
                self._dlg_tbl_open.setItem(row, 3, QTableWidgetItem(f"{pos.price_open:.5f}"))
                self._dlg_tbl_open.setItem(row, 4, QTableWidgetItem(f"{pos.price_current:.5f}"))
                pnl_item = QTableWidgetItem(f"${pnl:.2f}")
                pnl_item.setForeground(QColor("#4ADE80") if pnl >= 0 else QColor("#F87171"))
                self._dlg_tbl_open.setItem(row, 5, pnl_item)
                self._dlg_tbl_open.setItem(row, 6, QTableWidgetItem(f"{pos.sl:.5f}"))
                self._dlg_tbl_open.setItem(row, 7, QTableWidgetItem(f"{pos.tp:.5f}"))

                # V4 Layers status
                tp_state = self._tp_manager._tp_state.get(pos.ticket, {})
                if tp_state:
                    layers = []
                    if tp_state.get("micro_tp"):
                        layers.append("uTP")
                    if tp_state.get("be_activated"):
                        layers.append("BE")
                    if tp_state.get("stall_be"):
                        layers.append("STALL")
                    if tp_state.get("tp1"):
                        layers.append("TP1")
                    if tp_state.get("tp2"):
                        layers.append("TP2")
                    if tp_state.get("tp3"):
                        layers.append("TP3")
                    if tp_state.get("profit_trail_sl") is not None:
                        layers.append("TRAIL")
                    layer_text = " ".join(layers) if layers else "---"
                else:
                    layer_text = "untracked"
                layer_item = QTableWidgetItem(layer_text)
                if "TP1" in layer_text:
                    layer_item.setForeground(QColor("#4ADE80"))
                elif "BE" in layer_text:
                    layer_item.setForeground(QColor("#FBBF24"))
                else:
                    layer_item.setForeground(QColor("#B5AFA5"))
                self._dlg_tbl_open.setItem(row, 8, layer_item)

                # Trail SL
                trail_sl = tp_state.get("profit_trail_sl") if tp_state else None
                trail_text = f"{trail_sl:.5f}" if trail_sl is not None else "---"
                self._dlg_tbl_open.setItem(row, 9, QTableWidgetItem(trail_text))

                # Peak (max favorable price)
                peak = tp_state.get("max_favorable") if tp_state else None
                peak_text = f"{peak:.5f}" if peak is not None else "---"
                self._dlg_tbl_open.setItem(row, 10, QTableWidgetItem(peak_text))

            # --- Trade history (today's deals) ---
            from datetime import datetime, timedelta
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            deals = mt5.history_deals_get(today_start, now)
            if deals is None:
                deals = []

            # Filter: only our magic, only out-deals (closes), not deposits/withdrawals
            close_deals = [d for d in deals
                           if d.magic == MAGIC_NUMBER
                           and d.entry == mt5.DEAL_ENTRY_OUT
                           and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]

            self._dlg_tbl_history.setRowCount(len(close_deals))
            for row, deal in enumerate(close_deals):
                direction = "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL"
                deal_time = datetime.fromtimestamp(deal.time).strftime("%H:%M:%S")
                self._dlg_tbl_history.setItem(row, 0, QTableWidgetItem(deal.symbol))
                self._dlg_tbl_history.setItem(row, 1, QTableWidgetItem(direction))
                self._dlg_tbl_history.setItem(row, 2, QTableWidgetItem(f"{deal.volume:.2f}"))
                self._dlg_tbl_history.setItem(row, 3, QTableWidgetItem(f"{deal.price:.5f}"))
                self._dlg_tbl_history.setItem(row, 4, QTableWidgetItem(f"{deal.price:.5f}"))
                pnl_item = QTableWidgetItem(f"${deal.profit:.2f}")
                pnl_item.setForeground(QColor("#4ADE80") if deal.profit >= 0 else QColor("#F87171"))
                self._dlg_tbl_history.setItem(row, 5, pnl_item)
                self._dlg_tbl_history.setItem(row, 6, QTableWidgetItem(f"${deal.commission:.2f}"))
                self._dlg_tbl_history.setItem(row, 7, QTableWidgetItem(deal_time))

        except Exception as e:
            pass

    def _on_scan_data(self, symbol, df):
        """Update lightweight-chart when scan data arrives."""
        self._chart_data[symbol] = df

        if symbol not in self.chart_objects:
            return

        chart = self.chart_objects[symbol]
        zp_lines = self.chart_lines.get(symbol)  # (bull_line, bear_line)

        try:
            # Prepare OHLCV for chart
            chart_df = df[["time", "open", "high", "low", "close", "volume"]].copy()

            # Split trailing stop into bull/bear segments by position
            bull_df = df[["time", "xATRTrailingStop", "pos"]].copy()
            bear_df = bull_df.copy()
            bull_df.loc[bull_df["pos"] != 1, "xATRTrailingStop"] = np.nan
            bear_df.loc[bear_df["pos"] != -1, "xATRTrailingStop"] = np.nan
            bull_df = bull_df[["time", "xATRTrailingStop"]].rename(columns={"xATRTrailingStop": "ZP Bull"})
            bear_df = bear_df[["time", "xATRTrailingStop"]].rename(columns={"xATRTrailingStop": "ZP Bear"})
            bull_df = bull_df.dropna(subset=["ZP Bull"])
            bear_df = bear_df.dropna(subset=["ZP Bear"])

            if not self._chart_loaded.get(symbol, False):
                # First load — set full historical data
                chart.set(chart_df)
                if zp_lines:
                    zp_bull, zp_bear = zp_lines
                    if len(bull_df) > 0:
                        zp_bull.set(bull_df)
                    if len(bear_df) > 0:
                        zp_bear.set(bear_df)

                # Add markers only on actual flip signals
                for idx, row in df.iterrows():
                    if row.get("buy_signal", False):
                        chart.marker(
                            time=row["time"],
                            position="below",
                            shape="arrow_up",
                            color="#4ADE80",
                            text="BUY",
                        )
                    elif row.get("sell_signal", False):
                        chart.marker(
                            time=row["time"],
                            position="above",
                            shape="arrow_down",
                            color="#F87171",
                            text="SELL",
                        )

                # Fit chart to show all data
                chart.fit()
                self._chart_loaded[symbol] = True
            else:
                # Incremental update — update last bar
                last_row = chart_df.iloc[-1]
                chart.update(last_row)

                if zp_lines:
                    zp_bull, zp_bear = zp_lines
                    if len(bull_df) > 0:
                        zp_bull.update(bull_df.iloc[-1])
                    if len(bear_df) > 0:
                        zp_bear.update(bear_df.iloc[-1])

                # Check for new signal on confirmed bar
                if len(df) > 1:
                    prev = df.iloc[-2]
                    if prev.get("buy_signal", False):
                        chart.marker(
                            time=prev["time"],
                            position="below",
                            shape="arrow_up",
                            color="#4ADE80",
                            text="BUY",
                        )
                    elif prev.get("sell_signal", False):
                        chart.marker(
                            time=prev["time"],
                            position="above",
                            shape="arrow_down",
                            color="#F87171",
                            text="SELL",
                        )

        except Exception as e:
            self._log(f"[{symbol}] Chart update error: {e}")

    # ── Logging ──

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}"
        log.info(msg)
        self._log_signal.emit(full)

    def _log_on_main_thread(self, msg):
        self.txt_log.append(msg)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Live account stats ──

    def _update_live(self):
        try:
            if not mt5.terminal_info():
                if not mt5.initialize():
                    return

            acct = mt5.account_info()
            if not acct:
                return

            balance = acct.balance
            equity = acct.equity
            margin = acct.margin
            free_margin = acct.margin_free

            # Store starting balance on first connect
            if self._starting_balance is None:
                self._starting_balance = balance

            # Update labels
            self.lbl_balance.setText(f"Bal: ${balance:.2f}")

            eq_color = "#4ADE80" if equity >= balance else "#F87171"
            self.lbl_equity.setText(f"Eq: ${equity:.2f}")
            self.lbl_equity.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {eq_color};")

            self.lbl_margin.setText(f"Margin: ${margin:.2f}")
            self.lbl_free_margin.setText(f"Free: ${free_margin:.2f}")

            # Growth % from starting balance
            if self._starting_balance > 0:
                growth = ((equity - self._starting_balance) / self._starting_balance) * 100
                g_color = "#4ADE80" if growth >= 0 else "#F87171"
                sign = "+" if growth >= 0 else ""
                self.lbl_growth.setText(f"Growth: {sign}{growth:.2f}%")
                self.lbl_growth.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {g_color};")

            # Update V4 manager active trade count
            managed = len(self._tp_manager._trade_info)
            be_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("be_activated"))
            tp1_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("tp1"))
            trail_count = sum(1 for s in self._tp_manager._tp_state.values() if s.get("profit_trail_sl") is not None)
            if managed > 0:
                parts = [f"{managed} managed"]
                if be_count:
                    parts.append(f"{be_count} BE")
                if tp1_count:
                    parts.append(f"{tp1_count} TP1")
                if trail_count:
                    parts.append(f"{trail_count} trailing")
                self.lbl_v4_active.setText("V4: " + " | ".join(parts))
                self.lbl_v4_active.setStyleSheet("font-weight: bold; font-size: 12px; color: #4ADE80;")
            else:
                self.lbl_v4_active.setText("V4: Waiting for trades...")
                self.lbl_v4_active.setStyleSheet("font-weight: bold; font-size: 12px; color: #6B6355;")

        except Exception:
            pass

    # ── Close All ──

    def _on_close_all(self):
        try:
            positions = mt5.positions_get()
            if not positions:
                self._log("No positions to close")
                return

            my_pos = [p for p in positions if p.magic == MAGIC_NUMBER]
            closed = 0
            for pos in my_pos:
                close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                sym_info = mt5.symbol_info(pos.symbol)
                if sym_info is None:
                    continue
                price = sym_info.bid if pos.type == 0 else sym_info.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": "ACi-CloseAll",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                for fill in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed += 1
                        break

            self._log(f"Closed {closed}/{len(my_pos)} positions")
        except Exception as e:
            self._log(f"Close all error: {e}")

    # ── Theme toggle ──

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        self.setStyleSheet(DARK_STYLE if self._dark_mode else CLAUDE_STYLE)
        self.btn_theme.setText("Light Mode" if self._dark_mode else "Dark Mode")

        # Update chart themes
        for sym, chart in self.chart_objects.items():
            try:
                if self._dark_mode:
                    chart.layout(background_color="#1A1815", text_color="#B5AFA5",
                                font_size=12, font_family="Georgia")
                else:
                    chart.layout(background_color="#FAF9F5", text_color="#3D3929",
                                font_size=12, font_family="Georgia")
            except Exception:
                pass

        self._save_settings()

    # ── Settings ──

    def _save_settings(self):
        settings = {
            "dark_mode": self._dark_mode,
            "risk": self.inp_risk.text(),
            "lots": self.inp_lots.text(),
            "max_trades": self.inp_max_trades.text(),
            "poll_sec": self.inp_poll.text(),
            "timeframe": self.combo_tf.currentData(),
            "pairs": {s: cb.isChecked() for s, cb in self.pair_checks.items()},
            # V4 Profit Capture parameters
            "v4_tp1": self.inp_tp1.text(),
            "v4_tp2": self.inp_tp2.text(),
            "v4_tp3": self.inp_tp3.text(),
            "v4_be_trigger": self.inp_be_trigger.text(),
            "v4_be_buffer": self.inp_be_buffer.text(),
            "v4_micro_tp": self.inp_micro_tp.text(),
            "v4_micro_pct": self.inp_micro_pct.text(),
            "v4_stall": self.inp_stall.text(),
            "v4_trail": self.inp_trail.text(),
        }
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, "r") as f:
                s = json.load(f)
            self.inp_risk.setText(s.get("risk", "30"))
            self.inp_lots.setText(s.get("lots", "0.40"))
            self.inp_max_trades.setText(s.get("max_trades", "5"))
            self.inp_poll.setText(s.get("poll_sec", "30"))
            saved_tf = s.get("timeframe", "H4")
            for i in range(self.combo_tf.count()):
                if self.combo_tf.itemData(i) == saved_tf:
                    self.combo_tf.setCurrentIndex(i)
                    break
            for sym, active in s.get("pairs", {}).items():
                if sym in self.pair_checks:
                    self.pair_checks[sym].setChecked(active)
            if not s.get("dark_mode", True):
                self._toggle_theme()
            # V4 Profit Capture parameters
            if "v4_tp1" in s:
                self.inp_tp1.setText(s["v4_tp1"])
            if "v4_tp2" in s:
                self.inp_tp2.setText(s["v4_tp2"])
            if "v4_tp3" in s:
                self.inp_tp3.setText(s["v4_tp3"])
            if "v4_be_trigger" in s:
                self.inp_be_trigger.setText(s["v4_be_trigger"])
            if "v4_be_buffer" in s:
                self.inp_be_buffer.setText(s["v4_be_buffer"])
            if "v4_micro_tp" in s:
                self.inp_micro_tp.setText(s["v4_micro_tp"])
            if "v4_micro_pct" in s:
                self.inp_micro_pct.setText(s["v4_micro_pct"])
            if "v4_stall" in s:
                self.inp_stall.setText(s["v4_stall"])
            if "v4_trail" in s:
                self.inp_trail.setText(s["v4_trail"])
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def closeEvent(self, event):
        self._on_stop()
        self._tp_manager.stop()
        self._save_settings()
        try:
            mt5.shutdown()
        except Exception:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = ACiApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

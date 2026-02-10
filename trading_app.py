#!/usr/bin/env python3
"""
Neural Forex Trader — PySide6 Production UI
"""

import sys, os, threading, logging, json, time as _time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Ensure app modules importable
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QLineEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QHeaderView, QSplitter, QFrame, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QColor, QIcon

# Safe imports — show clear errors instead of silent crashes
_IMPORT_ERRORS = []

try:
    import MetaTrader5 as _mt5_check
except ImportError:
    _IMPORT_ERRORS.append("MetaTrader5 not installed. Run: pip install MetaTrader5")

try:
    from app.trading_engine import TradingEngine
    from app.model_manager import NeuralModelManager
    from app.mt5_connector import MT5Connector
    from app.config_manager import ConfigManager
except ImportError as e:
    _IMPORT_ERRORS.append(f"Core module import failed: {e}")

try:
    from agentic_orchestrator import AgenticOrchestrator
except ImportError:
    AgenticOrchestrator = None  # Optional — trading works without it

# ---------------------------------------------------------------------------
# Dark theme stylesheet
# ---------------------------------------------------------------------------
DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #333355;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #8888cc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #2a2a4a;
    border: 1px solid #444477;
    border-radius: 4px;
    padding: 6px 16px;
    color: #e0e0e0;
    font-weight: bold;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #3a3a5a;
    border-color: #6666aa;
}
QPushButton:pressed {
    background-color: #1a1a3a;
}
QPushButton:disabled {
    background-color: #1a1a2e;
    color: #555;
    border-color: #333;
}
QPushButton#startBtn {
    background-color: #1a4a2a;
    border-color: #2a7a3a;
}
QPushButton#startBtn:hover {
    background-color: #2a5a3a;
}
QPushButton#stopBtn {
    background-color: #4a2a1a;
    border-color: #7a3a2a;
}
QPushButton#stopBtn:hover {
    background-color: #5a3a2a;
}
QPushButton#emergencyBtn {
    background-color: #6a1a1a;
    border-color: #aa2222;
    color: #ff6666;
}
QPushButton#emergencyBtn:hover {
    background-color: #7a2a2a;
}
QLineEdit {
    background-color: #0f0f23;
    border: 1px solid #333355;
    border-radius: 3px;
    padding: 4px 8px;
    color: #e0e0e0;
}
QLineEdit:focus {
    border-color: #6666aa;
}
QCheckBox {
    spacing: 6px;
    color: #e0e0e0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #555;
    border-radius: 3px;
    background-color: #0f0f23;
}
QCheckBox::indicator:checked {
    background-color: #4466cc;
    border-color: #6688ee;
}
QTableWidget {
    background-color: #0f0f23;
    border: 1px solid #333355;
    border-radius: 4px;
    gridline-color: #222244;
    selection-background-color: #2a2a5a;
    color: #e0e0e0;
}
QTableWidget::item {
    padding: 4px;
}
QHeaderView::section {
    background-color: #1a1a3e;
    color: #8888cc;
    border: none;
    border-bottom: 1px solid #333355;
    padding: 6px;
    font-weight: bold;
}
QTextEdit {
    background-color: #0a0a1a;
    border: 1px solid #333355;
    border-radius: 4px;
    color: #88cc88;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}
QLabel#statusLabel {
    font-size: 12px;
    padding: 2px 8px;
}
QLabel#balanceLabel {
    font-size: 16px;
    font-weight: bold;
    color: #66ccff;
}
QFrame#separator {
    background-color: #333355;
    max-height: 1px;
}
"""

ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD',
]


class TradingApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Forex Trader")
        self.setMinimumSize(780, 620)
        self.resize(780, 660)

        self._setup_logging()

        # Core components
        self.config_manager = ConfigManager()
        self.mt5_connector = MT5Connector()
        self.model_manager = NeuralModelManager()
        self.trading_engine: Optional[TradingEngine] = None
        self.orchestrator = None
        self.is_trading = False

        self._build_ui()

        # Live update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_live)
        self.timer.start(2000)

        # Show import errors if any
        if _IMPORT_ERRORS:
            for err in _IMPORT_ERRORS:
                self._log(f"<span style='color:#ff4444'>⚠ {err}</span>")
            self._log("<span style='color:#ffcc44'>Fix the above errors then restart the app.</span>")
        else:
            self._log("Ready. Connect MT5 and load model to begin.")

    # ------------------------------------------------------------------
    def _setup_logging(self):
        Path("logs").mkdir(exist_ok=True)
        self.logger = logging.getLogger("TradingApp")
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler("logs/trading.log")
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(fh)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(12, 10, 12, 10)

        # ---- Status bar ----
        status_row = QHBoxLayout()
        self.lbl_mt5 = QLabel("MT5: --")
        self.lbl_mt5.setObjectName("statusLabel")
        self.lbl_model = QLabel("Model: --")
        self.lbl_model.setObjectName("statusLabel")
        self.lbl_trading = QLabel("Trading: OFF")
        self.lbl_trading.setObjectName("statusLabel")
        self.lbl_balance = QLabel("$0.00")
        self.lbl_balance.setObjectName("balanceLabel")
        self.lbl_balance.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for lbl in [self.lbl_mt5, self.lbl_model, self.lbl_trading]:
            status_row.addWidget(lbl)
        status_row.addStretch()
        status_row.addWidget(self.lbl_balance)
        root.addLayout(status_row)

        # Separator
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.HLine)
        root.addWidget(sep)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("Connect MT5")
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self._on_load_model)
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_emergency = QPushButton("Emergency Stop")
        self.btn_emergency.setObjectName("emergencyBtn")
        self.btn_emergency.clicked.connect(self._on_emergency)

        for b in [self.btn_connect, self.btn_load, self.btn_start, self.btn_stop]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_emergency)
        root.addLayout(btn_row)

        # ---- Settings (two group boxes side by side) ----
        settings_row = QHBoxLayout()

        # Left: Mode
        mode_box = QGroupBox("Mode")
        mode_grid = QGridLayout(mode_box)
        mode_grid.setSpacing(6)

        self.chk_zp = QCheckBox("ZeroPoint Pure Mode")
        mode_grid.addWidget(self.chk_zp, 0, 0, 1, 2)

        mode_grid.addWidget(QLabel("Fixed Lots:"), 1, 0)
        self.inp_lot = QLineEdit("0.40")
        self.inp_lot.setFixedWidth(70)
        mode_grid.addWidget(self.inp_lot, 1, 1)

        mode_grid.addWidget(QLabel("Risk %:"), 2, 0)
        self.inp_risk = QLineEdit("8")
        self.inp_risk.setFixedWidth(70)
        mode_grid.addWidget(self.inp_risk, 2, 1)

        mode_grid.addWidget(QLabel("Confidence %:"), 3, 0)
        self.inp_conf = QLineEdit("65")
        self.inp_conf.setFixedWidth(70)
        mode_grid.addWidget(self.inp_conf, 3, 1)

        settings_row.addWidget(mode_box)

        # Right: Trade Monitor
        monitor_box = QGroupBox("Trade Monitor")
        mon_grid = QGridLayout(monitor_box)
        mon_grid.setSpacing(6)

        mon_grid.addWidget(QLabel("Max Loss ($):"), 0, 0)
        self.inp_maxloss = QLineEdit("80")
        self.inp_maxloss.setFixedWidth(70)
        mon_grid.addWidget(self.inp_maxloss, 0, 1)

        mon_grid.addWidget(QLabel("BE Pips:"), 1, 0)
        self.inp_be = QLineEdit("15")
        self.inp_be.setFixedWidth(70)
        mon_grid.addWidget(self.inp_be, 1, 1)

        mon_grid.addWidget(QLabel("Stall (min):"), 2, 0)
        self.inp_stall = QLineEdit("30")
        self.inp_stall.setFixedWidth(70)
        mon_grid.addWidget(self.inp_stall, 2, 1)

        mon_grid.addWidget(QLabel("Deadline (min):"), 3, 0)
        self.inp_deadline = QLineEdit("60")
        self.inp_deadline.setFixedWidth(70)
        mon_grid.addWidget(self.inp_deadline, 3, 1)

        mon_grid.addWidget(QLabel("Profit Target ($):"), 4, 0)
        self.inp_profit_target = QLineEdit("")
        self.inp_profit_target.setFixedWidth(70)
        self.inp_profit_target.setPlaceholderText("e.g. 120")
        self.inp_profit_target.setStyleSheet(
            "background: #0f0f23; border: 1px solid #446644; color: #44ff88; padding: 4px 8px;")
        mon_grid.addWidget(self.inp_profit_target, 4, 1)

        settings_row.addWidget(monitor_box)
        root.addLayout(settings_row)

        # ---- Pairs row ----
        pairs_box = QGroupBox("Pairs")
        pairs_row = QHBoxLayout(pairs_box)
        pairs_row.setSpacing(10)
        self.pair_checks = {}
        for pair in ALL_PAIRS:
            chk = QCheckBox(pair)
            chk.setChecked(pair != 'USDJPY')
            self.pair_checks[pair] = chk
            pairs_row.addWidget(chk)
        pairs_row.addStretch()
        root.addWidget(pairs_box)

        # ---- Positions table ----
        pos_box = QGroupBox("Open Positions")
        pos_layout = QVBoxLayout(pos_box)
        pos_layout.setContentsMargins(4, 14, 4, 4)

        # Total P/L + Profit Target display row
        pnl_row = QHBoxLayout()
        self.lbl_total_pnl = QLabel("Total P/L: $0.00")
        self.lbl_total_pnl.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #e0e0e0; padding: 2px 4px;")
        pnl_row.addWidget(self.lbl_total_pnl)
        self.lbl_target_status = QLabel("")
        self.lbl_target_status.setStyleSheet(
            "font-size: 12px; color: #888; padding: 2px 4px;")
        pnl_row.addWidget(self.lbl_target_status)
        pnl_row.addStretch()
        self.btn_close_all = QPushButton("Close All")
        self.btn_close_all.setFixedWidth(80)
        self.btn_close_all.setStyleSheet("background-color: #4a2a1a; border-color: #7a3a2a;")
        self.btn_close_all.clicked.connect(self._on_close_all)
        pnl_row.addWidget(self.btn_close_all)
        pos_layout.addLayout(pnl_row)

        # TP controls row
        tp_row = QHBoxLayout()
        tp_row.addWidget(QLabel("Set All TP:"))
        self.inp_all_tp = QLineEdit()
        self.inp_all_tp.setFixedWidth(100)
        self.inp_all_tp.setPlaceholderText("e.g. 1.35000")
        tp_row.addWidget(self.inp_all_tp)
        self.btn_set_all_tp = QPushButton("Apply to All")
        self.btn_set_all_tp.setFixedWidth(100)
        self.btn_set_all_tp.clicked.connect(self._on_set_all_tp)
        tp_row.addWidget(self.btn_set_all_tp)
        tp_row.addStretch()
        pos_layout.addLayout(tp_row)

        self.pos_table = QTableWidget(0, 11)
        self.pos_table.setHorizontalHeaderLabels(
            ['Symbol', 'Dir', 'Lots', 'Entry', 'Current', 'P/L', 'Timer', 'SL', 'TP', 'New TP', ''])
        header = self.pos_table.horizontalHeader()
        for col in range(9):  # Symbol..TP stretch
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        header.setSectionResizeMode(9, QHeaderView.Fixed)   # New TP edit
        self.pos_table.setColumnWidth(9, 90)
        header.setSectionResizeMode(10, QHeaderView.Fixed)   # Set button
        self.pos_table.setColumnWidth(10, 70)
        self.pos_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pos_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pos_table.verticalHeader().setVisible(False)
        self.pos_table.setMaximumHeight(180)

        # Store position data for actions
        self._pos_data = []

        pos_layout.addWidget(self.pos_table)
        root.addWidget(pos_box)

        # ---- Log ----
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(4, 14, 4, 4)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        log_layout.addWidget(self.log_view)
        root.addWidget(log_box)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_view.append(f"<span style='color:#555'>[{ts}]</span> {msg}")
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _set_status(self, label, text, ok):
        color = "#44cc66" if ok else "#666"
        label.setText(text)
        label.setStyleSheet(f"color: {color};")

    def _float(self, widget, default):
        try:
            return float(widget.text())
        except ValueError:
            return default

    @staticmethod
    def _fmt_duration(open_timestamp):
        """Format seconds since open_timestamp as  Xh Ym  or  Ym Zs."""
        elapsed = int(_time.time() - open_timestamp)
        if elapsed < 0:
            elapsed = 0
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        elif m > 0:
            return f"{m}m {s:02d}s"
        else:
            return f"{s}s"

    # ------------------------------------------------------------------
    # Live update
    # ------------------------------------------------------------------
    def _update_live(self):
        mt5_ok = self.mt5_connector.is_connected()
        model_ok = self.model_manager.is_model_loaded()

        self._set_status(self.lbl_mt5, f"MT5: {'ON' if mt5_ok else 'OFF'}", mt5_ok)
        self._set_status(self.lbl_model, f"Model: {'OK' if model_ok else '--'}", model_ok)
        self._set_status(self.lbl_trading,
                         f"Trading: {'LIVE' if self.is_trading else 'OFF'}",
                         self.is_trading)

        if mt5_ok:
            try:
                info = self.mt5_connector.get_account_info()
                if info:
                    bal = float(info.get('balance', 0))
                    eq = float(info.get('equity', 0))
                    self.lbl_balance.setText(f"${bal:,.2f}  (eq: ${eq:,.2f})")
            except Exception:
                pass

        if mt5_ok and not self.is_trading:
            self.btn_start.setEnabled(True)

        self._refresh_positions()

    def _refresh_positions(self):
        if not self.mt5_connector.is_connected():
            self.pos_table.setRowCount(0)
            self._pos_data = []
            self.lbl_total_pnl.setText("Total P/L: $0.00")
            self.lbl_total_pnl.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #e0e0e0; padding: 2px 4px;")
            self.lbl_target_status.setText("")
            return
        try:
            import MetaTrader5 as mt5_lib
            positions = mt5_lib.positions_get()
            if not positions:
                self.pos_table.setRowCount(0)
                self._pos_data = []
                self.lbl_total_pnl.setText("Total P/L: $0.00")
                self.lbl_total_pnl.setStyleSheet(
                    "font-size: 15px; font-weight: bold; color: #e0e0e0; padding: 2px 4px;")
                self.lbl_target_status.setText("")
                return

            # Only rebuild table if position count changed (avoids flickering edits)
            if len(positions) != len(self._pos_data) or \
               [p.ticket for p in positions] != [d['ticket'] for d in self._pos_data]:
                self._rebuild_positions_table(positions)
            else:
                # Just update price / P/L columns (don't touch TP edit fields)
                self._update_positions_values(positions)

            # --- Total P/L + Profit Target check ---
            total_pnl = sum(p.profit for p in positions)
            pnl_color = "#44ff88" if total_pnl >= 0 else "#ff4444"
            self.lbl_total_pnl.setText(f"Total P/L: ${total_pnl:+.2f}")
            self.lbl_total_pnl.setStyleSheet(
                f"font-size: 15px; font-weight: bold; color: {pnl_color}; padding: 2px 4px;")

            # Check profit target
            target_text = self.inp_profit_target.text().strip()
            if target_text:
                try:
                    target = float(target_text)
                    pct = (total_pnl / target * 100) if target > 0 else 0
                    self.lbl_target_status.setText(
                        f"Target: ${target:.0f}  ({pct:.0f}%)")
                    if total_pnl >= target:
                        self.lbl_target_status.setStyleSheet(
                            "font-size: 12px; color: #44ff88; font-weight: bold; padding: 2px 4px;")
                        self._log(
                            f"<span style='color:#44ff88'>💰 PROFIT TARGET HIT! "
                            f"${total_pnl:+.2f} >= ${target:.0f} — Closing all positions</span>")
                        self.inp_profit_target.setText("")  # Clear to prevent re-trigger
                        self._on_close_all()
                    else:
                        self.lbl_target_status.setStyleSheet(
                            "font-size: 12px; color: #aaaacc; padding: 2px 4px;")
                except ValueError:
                    self.lbl_target_status.setText("(invalid target)")
                    self.lbl_target_status.setStyleSheet(
                        "font-size: 12px; color: #cc6644; padding: 2px 4px;")
            else:
                self.lbl_target_status.setText("")

        except Exception:
            pass

    def _rebuild_positions_table(self, positions):
        """Full rebuild of positions table with edit fields and buttons."""
        self.pos_table.setRowCount(len(positions))
        self._pos_data = []

        for i, p in enumerate(positions):
            direction = "BUY" if p.type == 0 else "SELL"
            pnl = p.profit
            color = QColor("#44cc66") if pnl >= 0 else QColor("#cc4444")
            timer_str = self._fmt_duration(p.time)

            pos_info = {
                'ticket': p.ticket, 'symbol': p.symbol,
                'direction': direction, 'volume': p.volume,
                'sl': p.sl, 'tp': p.tp,
            }
            self._pos_data.append(pos_info)

            vals = [
                p.symbol, direction, f"{p.volume:.2f}",
                f"{p.price_open:.5f}", f"{p.price_current:.5f}",
                f"${pnl:+.2f}", timer_str, f"{p.sl:.5f}", f"{p.tp:.5f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 5:  # P/L color
                    item.setForeground(color)
                if j == 6:  # Timer color
                    item.setForeground(QColor("#aaaaee"))
                self.pos_table.setItem(i, j, item)

            # New TP edit field (col 9)
            tp_edit = QLineEdit(f"{p.tp:.5f}")
            tp_edit.setAlignment(Qt.AlignCenter)
            tp_edit.setStyleSheet("background: #0f0f23; border: 1px solid #444; color: #ffcc44; padding: 2px;")
            self.pos_table.setCellWidget(i, 9, tp_edit)

            # Set TP button (col 10)
            btn = QPushButton("Set")
            btn.setStyleSheet("background: #2a4a2a; color: #88ff88; padding: 2px 8px; font-size: 11px;")
            row_idx = i
            btn.clicked.connect(lambda checked, r=row_idx: self._on_set_single_tp(r))
            self.pos_table.setCellWidget(i, 10, btn)

    def _update_positions_values(self, positions):
        """Update only price/P/L/timer columns without rebuilding edit fields."""
        for i, p in enumerate(positions):
            pnl = p.profit
            color = QColor("#44cc66") if pnl >= 0 else QColor("#cc4444")

            # Update Current price (col 4)
            item = QTableWidgetItem(f"{p.price_current:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 4, item)

            # Update P/L (col 5)
            item = QTableWidgetItem(f"${pnl:+.2f}")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(color)
            self.pos_table.setItem(i, 5, item)

            # Update Timer (col 6)
            item = QTableWidgetItem(self._fmt_duration(p.time))
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#aaaaee"))
            self.pos_table.setItem(i, 6, item)

            # Update SL (col 7) — may have been trailed
            item = QTableWidgetItem(f"{p.sl:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 7, item)

            # Update current TP display (col 8)
            item = QTableWidgetItem(f"{p.tp:.5f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.pos_table.setItem(i, 8, item)

            # Update stored data
            if i < len(self._pos_data):
                self._pos_data[i]['sl'] = p.sl
                self._pos_data[i]['tp'] = p.tp

    def _on_set_single_tp(self, row):
        """Set TP for a single position from the edit field in that row."""
        if row >= len(self._pos_data):
            return
        tp_widget = self.pos_table.cellWidget(row, 9)
        if tp_widget is None:
            return
        try:
            new_tp = float(tp_widget.text())
        except ValueError:
            self._log("Invalid TP value")
            return

        pos = self._pos_data[row]
        self._modify_tp(pos['ticket'], pos['symbol'], pos['sl'], new_tp)

    def _on_set_all_tp(self):
        """Set TP for ALL open positions from the global input."""
        text = self.inp_all_tp.text().strip()
        if not text:
            self._log("Enter a TP value first")
            return
        try:
            new_tp = float(text)
        except ValueError:
            self._log("Invalid TP value")
            return

        for pos in self._pos_data:
            self._modify_tp(pos['ticket'], pos['symbol'], pos['sl'], new_tp)

    def _on_close_all(self):
        """Close all open positions."""
        if not self._pos_data:
            return
        for pos in self._pos_data:
            self._close_position_mt5(pos['ticket'], pos['symbol'], pos['direction'], pos['volume'])

    def _modify_tp(self, ticket, symbol, sl, new_tp):
        """Send SLTP modify to MT5."""
        try:
            import MetaTrader5 as mt5_lib
            request = {
                "action": mt5_lib.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": sl,
                "tp": new_tp,
            }
            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(f"TP updated: {symbol} #{ticket} -> {new_tp:.5f}")
            else:
                rc = result.retcode if result else "?"
                self._log(f"TP update failed {symbol}: retcode={rc}")
        except Exception as e:
            self._log(f"TP error: {e}")

    def _close_position_mt5(self, ticket, symbol, direction, volume):
        """Close a position on MT5."""
        try:
            import MetaTrader5 as mt5_lib
            sym_info = mt5_lib.symbol_info(symbol)
            if not sym_info:
                return
            if direction == "BUY":
                close_type = mt5_lib.ORDER_TYPE_SELL
                price = sym_info.bid
            else:
                close_type = mt5_lib.ORDER_TYPE_BUY
                price = sym_info.ask

            request = {
                "action": mt5_lib.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "manual-close",
                "type_time": mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_FOK,
            }
            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(f"Closed: {symbol} {direction} #{ticket}")
            else:
                # Try other fill modes
                for fill in [mt5_lib.ORDER_FILLING_IOC, mt5_lib.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5_lib.order_send(request)
                    if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                        self._log(f"Closed: {symbol} {direction} #{ticket}")
                        return
                self._log(f"Close failed {symbol}: {result}")
        except Exception as e:
            self._log(f"Close error: {e}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_connect(self):
        def _work():
            try:
                self.btn_connect.setEnabled(False)
                self.btn_connect.setText("Connecting...")

                # Check if MetaTrader5 package is even installed
                try:
                    import MetaTrader5 as mt5_lib
                except ImportError:
                    self._log("<span style='color:#ff4444'>MetaTrader5 package not installed!</span>")
                    self._log("Run: <b>pip install MetaTrader5</b>")
                    return

                ok = self.mt5_connector.connect()
                if ok:
                    self._log("<span style='color:#44cc66'>MT5 connected</span>")
                    info = self.mt5_connector.get_account_info()
                    if info:
                        self._log(f"Account: {info.get('login','?')} | "
                                  f"Server: {info.get('server','?')} | "
                                  f"Balance: ${info.get('balance','?')}")
                else:
                    # Get the actual error from MT5
                    try:
                        err = mt5_lib.last_error()
                        err_code, err_msg = err if err else (0, "unknown")
                    except Exception:
                        err_code, err_msg = 0, "unknown"

                    self._log(f"<span style='color:#ff4444'>MT5 connection failed</span>")
                    self._log(f"Error: [{err_code}] {err_msg}")

                    # Give helpful advice based on error code
                    if err_code == -10003 or "IPC" in str(err_msg):
                        self._log("<span style='color:#ffcc44'>→ MT5 terminal is not running. Open MetaTrader 5 first.</span>")
                    elif err_code == -10004:
                        self._log("<span style='color:#ffcc44'>→ MT5 not found. Install MetaTrader 5 from your broker.</span>")
                    elif err_code == -10005:
                        self._log("<span style='color:#ffcc44'>→ Wrong MT5 version. Update MetaTrader 5.</span>")
                    elif "timeout" in str(err_msg).lower():
                        self._log("<span style='color:#ffcc44'>→ Connection timed out. Check your internet.</span>")
                    else:
                        self._log("<span style='color:#ffcc44'>→ Make sure MT5 is open and logged into an account.</span>")

            except Exception as e:
                self._log(f"<span style='color:#ff4444'>MT5 error: {e}</span>")
            finally:
                self.btn_connect.setEnabled(True)
                self.btn_connect.setText("Connect MT5")
        threading.Thread(target=_work, daemon=True).start()

    def _on_load_model(self):
        def _work():
            try:
                self.btn_load.setEnabled(False)
                self.btn_load.setText("Loading...")

                # Try multiple paths — ZeroPoint model first (best), then standard
                search_paths = [
                    'zeropoint_neural_model.pth',
                    'neural_model.pth',
                    'models/zeropoint_neural_model.pth',
                    'models/neural_model.pth',
                ]
                cfg_path = self.config_manager.get_config(
                    'trading', 'neural_network.model_path', '')
                if cfg_path and cfg_path not in search_paths:
                    search_paths.insert(0, cfg_path)

                ok = False
                used_path = None
                for p in search_paths:
                    if os.path.exists(p):
                        ok = self.model_manager.load_model(p)
                        if ok:
                            used_path = p
                            break

                if ok:
                    meta = getattr(self.model_manager, 'model_metadata', {}) or {}
                    dim = meta.get('feature_dim', '?')
                    name = os.path.basename(used_path)
                    self._log(f"Model loaded: {name} (features={dim})")
                else:
                    self._log("Model load failed - no .pth file found")
            except Exception as e:
                self._log(f"Model error: {e}")
            finally:
                self.btn_load.setEnabled(True)
                self.btn_load.setText("Load Model")
        threading.Thread(target=_work, daemon=True).start()

    def _on_start(self):
        """Start the ZP trade scanner — uses exact zp_trade_now.py logic."""
        try:
            if not self.mt5_connector.is_connected():
                self._log("Connect MT5 first")
                return

            self._zp_running = True
            self.is_trading = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

            lot = self._float(self.inp_lot, 0.40)
            self._log(f"ZP Scanner started | Lot: {lot} | Scanning every 60s for fresh H4 flips")

            def _zp_loop():
                import MetaTrader5 as mt5_lib
                import numpy as np
                import pandas as pd
                from app.zeropoint_signal import (
                    ZeroPointEngine, compute_zeropoint_state,
                    ZEROPOINT_ENABLED_SYMBOLS, SL_BUFFER_PCT, TP1_MULT,
                )

                zp_engine = ZeroPointEngine()

                while getattr(self, '_zp_running', False):
                    try:
                        selected = [p for p, c in self.pair_checks.items() if c.isChecked()]
                        fixed_lot = self._float(self.inp_lot, 0.40)

                        # Skip if we already have open positions
                        open_positions = mt5_lib.positions_get()
                        open_symbols = set()
                        if open_positions:
                            for pos in open_positions:
                                open_symbols.add(pos.symbol.upper().replace(".", "").replace("#", ""))

                        best_signal = None
                        best_symbol_resolved = None

                        for symbol in selected:
                            norm = symbol.upper().replace(".", "").replace("#", "")
                            if norm in open_symbols or symbol in open_symbols:
                                continue

                            # Resolve symbol on broker
                            sym_info = mt5_lib.symbol_info(symbol)
                            sym_resolved = symbol
                            if sym_info is None:
                                for alt in [symbol, symbol + ".raw", symbol[:3]]:
                                    sym_info = mt5_lib.symbol_info(alt)
                                    if sym_info is not None:
                                        sym_resolved = alt
                                        break
                            if sym_info is None:
                                continue

                            mt5_lib.symbol_select(sym_resolved, True)

                            rates_h4 = mt5_lib.copy_rates_from_pos(sym_resolved, mt5_lib.TIMEFRAME_H4, 0, 200)
                            rates_h1 = mt5_lib.copy_rates_from_pos(sym_resolved, mt5_lib.TIMEFRAME_H1, 0, 200)

                            df_h4 = None
                            if rates_h4 is not None and len(rates_h4) >= 20:
                                df_h4 = pd.DataFrame(rates_h4)
                                df_h4["time"] = pd.to_datetime(df_h4["time"], unit="s")
                            df_h1 = None
                            if rates_h1 is not None and len(rates_h1) >= 20:
                                df_h1 = pd.DataFrame(rates_h1)
                                df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
                            if df_h4 is None:
                                continue

                            # Try standard ZP engine, then raw fallback
                            sig = None
                            if norm in ZEROPOINT_ENABLED_SYMBOLS:
                                sig = zp_engine.generate_signal(sym_resolved, df_h4, df_h1)
                            if sig is None:
                                sig = self._zp_signal_fresh_only(sym_resolved, df_h4, df_h1)
                            if sig is None:
                                continue

                            self._log(
                                f"  {symbol}: ZP {sig.direction} "
                                f"R:R={sig.risk_reward:.2f} conf={sig.confidence:.0%} tier={sig.tier}"
                            )
                            if best_signal is None or sig.confidence > best_signal.confidence:
                                best_signal = sig
                                best_symbol_resolved = sym_resolved

                        if best_signal is not None:
                            self._zp_place_trade(best_signal, best_symbol_resolved, fixed_lot)
                        else:
                            self._log("Scan: no fresh H4 flip found")

                    except Exception as e:
                        self._log(f"Scan error: {e}")

                    # Wait 60s between scans (H4 data, no need to spam)
                    for _ in range(60):
                        if not getattr(self, '_zp_running', False):
                            break
                        _time.sleep(1)

            threading.Thread(target=_zp_loop, daemon=True).start()

        except Exception as e:
            self._log(f"Start error: {e}")

    @staticmethod
    def _zp_signal_fresh_only(symbol, df_h4, df_h1=None):
        """Exact zp_trade_now.py logic — FRESH FLIPS ONLY."""
        import numpy as np
        from app.zeropoint_signal import (
            compute_zeropoint_state, ZeroPointSignal,
            SL_BUFFER_PCT, TP1_MULT,
        )
        from datetime import datetime

        zp = compute_zeropoint_state(df_h4)
        if zp is None or len(zp) < 2:
            return None

        last = zp.iloc[-1]
        pos = int(last.get("pos", 0))
        if pos == 0:
            return None

        direction = "BUY" if pos == 1 else "SELL"

        # FRESH FLIP ONLY — current bar or 1-bar-ago
        buy_sig = bool(last.get("buy_signal", False))
        sell_sig = bool(last.get("sell_signal", False))
        is_fresh = buy_sig or sell_sig
        if not is_fresh:
            prev = zp.iloc[-2]
            is_fresh = bool(prev.get("buy_signal", False)) or bool(prev.get("sell_signal", False))
        if not is_fresh:
            return None

        entry = float(last["close"])
        atr_val = float(last["atr"])
        if atr_val <= 0 or np.isnan(atr_val):
            return None

        trailing_stop = float(last.get("xATRTrailingStop", 0))
        if trailing_stop <= 0:
            return None

        sl = trailing_stop
        buffer = atr_val * SL_BUFFER_PCT
        if direction == "BUY":
            sl = sl - buffer
            tp1 = entry + atr_val * TP1_MULT
        else:
            sl = sl + buffer
            tp1 = entry - atr_val * TP1_MULT

        if direction == "BUY" and entry >= tp1:
            return None
        if direction == "SELL" and entry <= tp1:
            return None

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp1 - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        # H1 confirmation
        h1_conf = False
        if df_h1 is not None:
            zp_h1 = compute_zeropoint_state(df_h1)
            if zp_h1 is not None and len(zp_h1) > 0:
                h1_pos = int(zp_h1.iloc[-1].get("pos", 0))
                if direction == "BUY" and h1_pos == 1:
                    h1_conf = True
                elif direction == "SELL" and h1_pos == -1:
                    h1_conf = True

        conf = 0.70 + (0.15 if h1_conf else 0.0) + min(rr * 0.05, 0.10)
        conf = max(0.50, min(conf, 0.98))
        tier = "S" if h1_conf else "A"

        return ZeroPointSignal(
            symbol=symbol, direction=direction, entry_price=entry,
            stop_loss=sl, tp1=tp1, tp2=tp1, tp3=tp1,
            atr_value=atr_val, confidence=conf,
            signal_time=datetime.now(), timeframe="H4",
            tier=tier, trailing_stop=trailing_stop,
            risk_reward=rr,
        )

    def _zp_place_trade(self, sig, sym_resolved, lot):
        """Place a trade exactly like zp_trade_now.py."""
        try:
            import MetaTrader5 as mt5_lib

            sym_info = mt5_lib.symbol_info(sym_resolved)
            if sym_info is None:
                self._log(f"Cannot get info for {sym_resolved}")
                return

            if sig.direction == "BUY":
                order_type = mt5_lib.ORDER_TYPE_BUY
                price = sym_info.ask
            else:
                order_type = mt5_lib.ORDER_TYPE_SELL
                price = sym_info.bid

            digits = sym_info.digits
            sl = round(sig.stop_loss, digits)
            tp = round(sig.tp1, digits)

            request = {
                "action": mt5_lib.TRADE_ACTION_DEAL,
                "symbol": sym_resolved,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": f"ZP-{sig.tier}",
                "type_time": mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_FOK,
            }

            result = mt5_lib.order_send(request)
            if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                self._log(
                    f"<span style='color:#44cc66'>TRADE: {sig.direction} {sym_resolved} "
                    f"@ {price:.5f} | SL={sl} TP={tp} | "
                    f"R:R={sig.risk_reward:.2f} Tier={sig.tier}</span>"
                )
            else:
                # Try other fill modes
                for fill in [mt5_lib.ORDER_FILLING_IOC, mt5_lib.ORDER_FILLING_RETURN]:
                    request["type_filling"] = fill
                    result = mt5_lib.order_send(request)
                    if result and result.retcode == mt5_lib.TRADE_RETCODE_DONE:
                        self._log(
                            f"<span style='color:#44cc66'>TRADE: {sig.direction} {sym_resolved} "
                            f"@ {price:.5f} | SL={sl} TP={tp}</span>"
                        )
                        return
                rc = result.retcode if result else "?"
                self._log(f"<span style='color:#ff4444'>Trade failed {sym_resolved}: {rc}</span>")

        except Exception as e:
            self._log(f"Trade error: {e}")

    def _on_stop(self):
        try:
            self._zp_running = False
            self.is_trading = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("Trading stopped")
        except Exception as e:
            self._log(f"Stop error: {e}")

    def _on_emergency(self):
        try:
            self._zp_running = False
            self.is_trading = False
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("<span style='color:#ff4444'>EMERGENCY STOP</span>")
        except Exception as e:
            self._log(f"Emergency error: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = TradingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

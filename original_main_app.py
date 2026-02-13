#!/usr/bin/env python3
"""
Neural Forex Trading App
========================

Cyber-style neural trading dashboard with live MT5 controls, model management,
trading telemetry, and an AI analyst corner panel.
"""

import os
import sys
import json
import shutil
import math
import random
import subprocess
import threading
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# Add app modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.trading_engine import TradingEngine
from app.model_manager import NeuralModelManager
from app.mt5_connector import MT5Connector
from app.config_manager import ConfigManager


class NeuralTradingApp:
    """Cyber-style neural forex trading application."""

    NAV_ITEMS = [
        ("dashboard", "Dashboard"),
        ("model", "Model Manager"),
        ("trading", "Trading Control"),
        ("logs", "Logs"),
        ("settings", "Settings"),
    ]

    DEFAULT_PAIRS = [
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

    COLORS = {
        "bg": "#060809",
        "panel": "#0B0F13",
        "panel_alt": "#11171D",
        "border": "#263340",
        "text": "#E8EEF2",
        "muted": "#90A0AE",
        "accent": "#00E6B8",
        "accent_alt": "#00B0FF",
        "danger": "#FF5C7A",
        "warn": "#FFC857",
        "success": "#2CFB9D",
        "button": "#1A222B",
        "button_active": "#24303B",
        "glow_low": "#133A35",
    }

    def __init__(self, root: tk.Tk):
        self.root = root

        # Core components
        self.config_manager = ConfigManager()
        self.model_manager = NeuralModelManager()
        self.mt5_connector = MT5Connector()
        self.trading_engine: Optional[TradingEngine] = None

        # State
        self.is_trading = False
        self.model_loaded = False
        self.mt5_connected = False
        self.current_page = "dashboard"
        self.latest_metrics: Dict[str, Any] = {}
        self.latest_signals: List[Dict[str, Any]] = []
        self.latest_positions: List[Dict[str, Any]] = []
        self.ai_messages: deque[str] = deque(maxlen=300)
        self.last_ai_emit_key = ""
        self.last_ai_emit_ts = datetime.min
        self.avatar_tick = 0
        self.avatar_expression = "neutral"
        self.avatar_target_expression = "neutral"
        self.last_total_pnl_for_face = 0.0
        self.last_total_trades_for_face = 0
        self.avatar_use_reference_face = False
        self.avatar_face_photo: Optional[tk.PhotoImage] = None
        self.avatar_face_bounds: Optional[tuple[int, int, int, int]] = None
        self.avatar_matrix_streams: List[Dict[str, Any]] = []
        self.binary_cells: List[Dict[str, Any]] = []
        self.binary_cols = 0
        self.binary_rows = 0
        self.binary_cell_size = 4
        self.binary_margin = (9, 8)
        self.matrix_heads: List[float] = []
        self.matrix_speeds: List[float] = []
        self.matrix_lengths: List[int] = []
        self.avatar_scanline = 0
        self.avatar_reference_image: Optional[tk.PhotoImage] = None
        self.avatar_reference_source: str = ""
        self.avatar_reference_target = Path("assets") / "binary_anime_reference.png"
        self.log_file_path = Path("logs") / "trading_app.log"
        self.log_file_offset = 0

        # Widget references
        self.nav_buttons: Dict[str, tk.Button] = {}
        self.pages: Dict[str, tk.Frame] = {}
        self.status_widgets: Dict[str, Dict[str, Any]] = {}
        self.account_value_labels: Dict[str, tk.Label] = {}
        self.perf_value_labels: Dict[str, tk.Label] = {}
        self.summary_value_labels: Dict[str, tk.Label] = {}
        self.risk_value_labels: Dict[str, tk.Label] = {}
        self.pair_vars: Dict[str, tk.BooleanVar] = {}

        self._configure_window()
        self._setup_logging()
        self._build_ttk_styles()
        self._create_gui()
        self._load_saved_settings_into_ui()
        self._initialize_log_offset()
        self._show_page("dashboard")
        self._queue_ai_message("AI analyst online. Monitoring symbols and execution health.")
        self._schedule_refresh()
        self._animate_avatar()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.logger.info("Neural Trading App initialized")

    def _configure_window(self) -> None:
        width = int(self.config_manager.get_config("main", "window.width", 1366))
        height = int(self.config_manager.get_config("main", "window.height", 820))
        self.root.title("Neural Trading App // Cyber Console")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1200, 760)
        self.root.configure(bg=self.COLORS["bg"])

    def _setup_logging(self) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("neural_trading_ui")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

        self.logger.propagate = False

    def _build_ttk_styles(self) -> None:
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure(
            "Cyber.Treeview",
            background=self.COLORS["panel"],
            foreground=self.COLORS["text"],
            fieldbackground=self.COLORS["panel"],
            bordercolor=self.COLORS["border"],
            borderwidth=1,
            rowheight=24,
            font=("Consolas", 10),
        )
        self.style.map(
            "Cyber.Treeview",
            background=[("selected", self.COLORS["accent_alt"])],
            foreground=[("selected", "#021014")],
        )
        self.style.configure(
            "Cyber.Treeview.Heading",
            background=self.COLORS["panel_alt"],
            foreground=self.COLORS["text"],
            relief="flat",
            font=("Bahnschrift SemiBold", 10),
        )
        self.style.map(
            "Cyber.Treeview.Heading",
            background=[("active", self.COLORS["button_active"])],
        )

    def _create_gui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._create_top_nav()
        self._create_page_host()
        self._create_bottom_actions()
        self._create_ai_panel()

    def _create_top_nav(self) -> None:
        top = tk.Frame(self.root, bg=self.COLORS["bg"], height=86)
        top.grid(row=0, column=0, sticky="ew", padx=22, pady=(14, 8))
        top.grid_columnconfigure(1, weight=1)

        title = tk.Label(
            top,
            text="System States",
            bg=self.COLORS["bg"],
            fg=self.COLORS["text"],
            font=("Bahnschrift SemiBold", 28),
        )
        title.grid(row=0, column=0, sticky="w", padx=(4, 28))

        nav = tk.Frame(top, bg=self.COLORS["bg"])
        nav.grid(row=0, column=1, sticky="e")

        for page_id, label in self.NAV_ITEMS:
            btn = tk.Button(
                nav,
                text=label,
                command=lambda p=page_id: self._show_page(p),
                bg=self.COLORS["button"],
                fg=self.COLORS["text"],
                activebackground=self.COLORS["button_active"],
                activeforeground=self.COLORS["text"],
                font=("Bahnschrift SemiBold", 13),
                relief="flat",
                bd=1,
                highlightthickness=1,
                highlightbackground=self.COLORS["border"],
                highlightcolor=self.COLORS["accent"],
                padx=20,
                pady=10,
                cursor="hand2",
            )
            btn.pack(side="left", padx=8)
            self.nav_buttons[page_id] = btn

    def _create_page_host(self) -> None:
        host = tk.Frame(self.root, bg=self.COLORS["bg"])
        host.grid(row=1, column=0, sticky="nsew", padx=22, pady=0)
        host.grid_rowconfigure(0, weight=1)
        host.grid_columnconfigure(0, weight=1)
        self.page_host = host

        self.pages["dashboard"] = self._create_dashboard_page(host)
        self.pages["model"] = self._create_model_page(host)
        self.pages["trading"] = self._create_trading_page(host)
        self.pages["logs"] = self._create_logs_page(host)
        self.pages["settings"] = self._create_settings_page(host)

        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _create_bottom_actions(self) -> None:
        bottom = tk.Frame(self.root, bg=self.COLORS["bg"], height=84)
        bottom.grid(row=2, column=0, sticky="ew", padx=22, pady=(10, 14))
        bottom.grid_columnconfigure(1, weight=1)

        divider = tk.Frame(bottom, bg=self.COLORS["border"], height=1)
        divider.pack(fill="x", pady=(0, 14))

        btn_row = tk.Frame(bottom, bg=self.COLORS["bg"])
        btn_row.pack(fill="x")

        self.connect_btn = self._make_action_button(
            btn_row, "Connect MT5", self.connect_mt5
        )
        self.connect_btn.pack(side="left", padx=(0, 12))

        self.load_model_btn = self._make_action_button(
            btn_row, "Load Model", self.load_model
        )
        self.load_model_btn.pack(side="left", padx=12)

        self.start_trading_btn = self._make_action_button(
            btn_row, "Start Trading", self.start_trading, accent=True
        )
        self.start_trading_btn.pack(side="left", padx=12)

        self.stop_trading_btn = self._make_action_button(
            btn_row, "Stop Trading", self.stop_trading
        )
        self.stop_trading_btn.configure(state="disabled")
        self.stop_trading_btn.pack(side="left", padx=12)

        self.bottom_status = tk.Label(
            btn_row,
            text="Console ready",
            bg=self.COLORS["bg"],
            fg=self.COLORS["muted"],
            font=("Consolas", 10),
        )
        self.bottom_status.pack(side="right")

    def _make_action_button(
        self, parent: tk.Widget, text: str, command, accent: bool = False
    ) -> tk.Button:
        bg = self.COLORS["accent"] if accent else self.COLORS["button"]
        fg = "#00130F" if accent else self.COLORS["text"]
        active_bg = "#00D6A9" if accent else self.COLORS["button_active"]
        active_fg = "#00130F" if accent else self.COLORS["text"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
            font=("Bahnschrift SemiBold", 13),
            relief="flat",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["accent"],
            padx=20,
            pady=9,
            cursor="hand2",
        )

    def _create_dashboard_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.COLORS["bg"])
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(1, weight=1)

        # Left system rail
        left = self._make_card(page, "System Status")
        left.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 14), pady=(0, 0))
        left.configure(width=360)
        left.grid_propagate(False)

        self._create_status_block(left, "mt5", "MT5 Connection")
        self._create_status_block(left, "model", "Neural Model")
        self._create_status_block(left, "trading", "Trading Engine")

        # Top cards
        top = tk.Frame(page, bg=self.COLORS["bg"])
        top.grid(row=0, column=1, sticky="nsew")
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)

        account_card = self._make_card(top, "Account")
        account_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        performance_card = self._make_card(top, "Performance")
        performance_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._create_label_value(account_card, "Account", "account")
        self._create_label_value(account_card, "Balance", "balance")
        self._create_label_value(account_card, "Equity", "equity")
        self._create_label_value(account_card, "Margin", "margin")
        self._create_label_value(account_card, "Free Margin", "free_margin")

        self._create_perf_value(performance_card, "Win Rate", "win_rate")
        self._create_perf_value(performance_card, "Total Trades", "total_trades")
        self._create_perf_value(performance_card, "Winning Trades", "winning_trades")
        self._create_perf_value(performance_card, "Daily P&L", "daily_pnl")
        self._create_perf_value(performance_card, "Total P&L", "total_pnl")
        
        # Risk monitoring section
        risk_card = self._make_card(top, "Risk Monitor")
        risk_card.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        
        self._create_risk_value(risk_card, "Current Drawdown", "current_drawdown")
        self._create_risk_value(risk_card, "Volatility Regime", "volatility_regime")
        self._create_risk_value(risk_card, "Risk Level", "risk_level")
        self._create_risk_value(risk_card, "Tail Risk Score", "tail_risk_score")
        self._create_risk_value(risk_card, "Dynamic Confidence", "dynamic_confidence")

        # Lower section
        bottom = tk.Frame(page, bg=self.COLORS["bg"])
        bottom.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
        bottom.grid_rowconfigure(1, weight=1)
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        summary = self._make_card(bottom, "Session Overview")
        summary.grid(row=0, column=0, columnspan=2, sticky="ew")
        summary.grid_columnconfigure(0, weight=1)
        summary.grid_columnconfigure(1, weight=1)
        summary.grid_columnconfigure(2, weight=1)

        self._create_summary_value(summary, "Win Rate", "sum_win_rate", 0)
        self._create_summary_value(summary, "Total Trades", "sum_total_trades", 1)
        self._create_summary_value(summary, "P&L", "sum_total_pnl", 2)

        radar = self._make_card(bottom, "Signal Radar")
        radar.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(14, 0))
        positions = self._make_card(bottom, "Open Positions")
        positions.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(14, 0))

        self.dashboard_signal_text = scrolledtext.ScrolledText(
            radar,
            height=10,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            wrap="word",
        )
        self.dashboard_signal_text.pack(fill="both", expand=True)

        self.dashboard_positions_text = scrolledtext.ScrolledText(
            positions,
            height=10,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            wrap="word",
        )
        self.dashboard_positions_text.pack(fill="both", expand=True)

        return page

    def _create_model_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.COLORS["bg"])
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        controls = self._make_card(page, "Model Controls")
        controls.grid(row=0, column=0, sticky="ew")

        btn_row = tk.Frame(controls, bg=self.COLORS["panel"])
        btn_row.pack(fill="x")
        self._make_action_button(btn_row, "Load Model", self.load_model).pack(
            side="left", padx=(0, 10)
        )
        self._make_action_button(btn_row, "Validate Model", self.validate_model).pack(
            side="left", padx=10
        )
        self._make_action_button(btn_row, "Train New Model", self.train_model).pack(
            side="left", padx=10
        )

        status_card = self._make_card(page, "Model Status")
        status_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.model_status_text = scrolledtext.ScrolledText(
            status_card,
            height=8,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            wrap="word",
        )
        self.model_status_text.pack(fill="both", expand=True)

        info_card = self._make_card(page, "Model Metadata")
        info_card.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        self.model_info_text = scrolledtext.ScrolledText(
            info_card,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            wrap="word",
        )
        self.model_info_text.pack(fill="both", expand=True)

        return page
    def _create_trading_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.COLORS["bg"])
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        settings = self._make_card(page, "Trading Parameters")
        settings.grid(row=0, column=0, sticky="ew")

        line1 = tk.Frame(settings, bg=self.COLORS["panel"])
        line1.pack(fill="x", pady=(0, 8))

        tk.Label(
            line1,
            text="Risk per Trade (%)",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Bahnschrift SemiBold", 11),
        ).pack(side="left")
        self.risk_var = tk.StringVar(value="1.5")
        tk.Entry(
            line1,
            textvariable=self.risk_var,
            bg=self.COLORS["panel_alt"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            width=8,
            font=("Consolas", 11),
        ).pack(side="left", padx=(10, 18))

        tk.Label(
            line1,
            text="Confidence Threshold (%)",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Bahnschrift SemiBold", 11),
        ).pack(side="left")
        self.confidence_var = tk.StringVar(value="65")
        tk.Entry(
            line1,
            textvariable=self.confidence_var,
            bg=self.COLORS["panel_alt"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            width=8,
            font=("Consolas", 11),
        ).pack(side="left", padx=(10, 18))

        tk.Label(
            line1,
            text="Max Positions",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Bahnschrift SemiBold", 11),
        ).pack(side="left")
        self.max_trades_var = tk.StringVar(value="5")
        tk.Entry(
            line1,
            textvariable=self.max_trades_var,
            bg=self.COLORS["panel_alt"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            width=8,
            font=("Consolas", 11),
        ).pack(side="left", padx=(10, 0))

        pairs_wrap = tk.Frame(settings, bg=self.COLORS["panel"])
        pairs_wrap.pack(fill="x")
        tk.Label(
            pairs_wrap,
            text="Active Symbols:",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Bahnschrift SemiBold", 11),
        ).pack(anchor="w")

        pair_grid = tk.Frame(pairs_wrap, bg=self.COLORS["panel"])
        pair_grid.pack(fill="x", pady=(6, 0))
        for i, symbol in enumerate(self.DEFAULT_PAIRS):
            var = tk.BooleanVar(value=True)
            self.pair_vars[symbol] = var
            check = tk.Checkbutton(
                pair_grid,
                text=symbol,
                variable=var,
                bg=self.COLORS["panel"],
                fg=self.COLORS["text"],
                activebackground=self.COLORS["panel"],
                activeforeground=self.COLORS["accent"],
                selectcolor=self.COLORS["panel_alt"],
                font=("Consolas", 10),
                relief="flat",
                bd=0,
                highlightthickness=0,
            )
            check.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 20), pady=2)

        signals = self._make_card(page, "Live Signals")
        signals.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.signals_tree = ttk.Treeview(
            signals,
            columns=("time", "symbol", "action", "confidence", "price", "reason"),
            show="headings",
            style="Cyber.Treeview",
            height=7,
        )
        self.signals_tree.heading("time", text="Time")
        self.signals_tree.heading("symbol", text="Symbol")
        self.signals_tree.heading("action", text="Action")
        self.signals_tree.heading("confidence", text="Confidence")
        self.signals_tree.heading("price", text="Entry")
        self.signals_tree.heading("reason", text="Reason")
        self.signals_tree.column("time", width=85, anchor="center")
        self.signals_tree.column("symbol", width=95, anchor="center")
        self.signals_tree.column("action", width=90, anchor="center")
        self.signals_tree.column("confidence", width=95, anchor="center")
        self.signals_tree.column("price", width=100, anchor="e")
        self.signals_tree.column("reason", width=360, anchor="w")
        self.signals_tree.pack(fill="both", expand=True)

        positions = self._make_card(page, "Active Positions")
        positions.grid(row=2, column=0, sticky="nsew", pady=(14, 0))
        self.positions_tree = ttk.Treeview(
            positions,
            columns=("ticket", "symbol", "side", "entry", "current", "pnl"),
            show="headings",
            style="Cyber.Treeview",
        )
        self.positions_tree.heading("ticket", text="Ticket")
        self.positions_tree.heading("symbol", text="Symbol")
        self.positions_tree.heading("side", text="Side")
        self.positions_tree.heading("entry", text="Entry")
        self.positions_tree.heading("current", text="Current")
        self.positions_tree.heading("pnl", text="Unrealized P&L")
        self.positions_tree.column("ticket", width=100, anchor="center")
        self.positions_tree.column("symbol", width=95, anchor="center")
        self.positions_tree.column("side", width=90, anchor="center")
        self.positions_tree.column("entry", width=115, anchor="e")
        self.positions_tree.column("current", width=115, anchor="e")
        self.positions_tree.column("pnl", width=150, anchor="e")
        self.positions_tree.pack(fill="both", expand=True)

        return page

    def _create_logs_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.COLORS["bg"])
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)

        controls = self._make_card(page, "Log Controls")
        controls.grid(row=0, column=0, sticky="ew")
        row = tk.Frame(controls, bg=self.COLORS["panel"])
        row.pack(fill="x")
        self._make_action_button(row, "Refresh Logs", self.refresh_logs).pack(
            side="left", padx=(0, 10)
        )
        self._make_action_button(row, "Clear Viewer", self.clear_logs).pack(
            side="left", padx=10
        )
        self._make_action_button(row, "Export Logs", self.export_logs).pack(
            side="left", padx=10
        )

        viewer = self._make_card(page, "Runtime Logs")
        viewer.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        self.log_text = scrolledtext.ScrolledText(
            viewer,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 10),
            wrap="none",
        )
        self.log_text.pack(fill="both", expand=True)

        self.refresh_logs()
        return page

    def _create_settings_page(self, parent: tk.Widget) -> tk.Frame:
        page = tk.Frame(parent, bg=self.COLORS["bg"])
        page.grid_columnconfigure(0, weight=1)

        mt5_card = self._make_card(page, "MT5 Connection Settings")
        mt5_card.grid(row=0, column=0, sticky="ew")

        self.server_var = tk.StringVar(value="auto")
        self.login_var = tk.StringVar(value="")
        self.password_var = tk.StringVar(value="")
        self.update_interval_var = tk.StringVar(value="5")

        self._create_entry_row(mt5_card, "Server", self.server_var)
        self._create_entry_row(mt5_card, "Login", self.login_var)
        self._create_entry_row(mt5_card, "Password", self.password_var, masked=True)

        app_card = self._make_card(page, "Application Settings")
        app_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self._create_entry_row(app_card, "Refresh Interval (sec)", self.update_interval_var)

        save_row = tk.Frame(page, bg=self.COLORS["bg"])
        save_row.grid(row=2, column=0, sticky="w", pady=(14, 0))
        self._make_action_button(save_row, "Save Settings", self.save_settings, accent=True).pack(
            side="left"
        )
        return page

    def _create_ai_panel(self) -> None:
        panel = tk.Frame(
            self.root,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            width=700,
            height=320,
        )
        panel.place(relx=1.0, rely=1.0, x=-20, y=-120, anchor="se")
        panel.pack_propagate(False)

        header = tk.Frame(panel, bg=self.COLORS["panel_alt"], height=34)
        header.pack(fill="x")
        tk.Label(
            header,
            text="AI Analyst // Mood & Trade Commentary",
            bg=self.COLORS["panel_alt"],
            fg="#F2F2F2",
            font=("Bahnschrift SemiBold", 11),
        ).pack(side="left", padx=10, pady=6)
        tk.Button(
            header,
            text="Load Face Ref",
            command=self.load_reference_image,
            bg=self.COLORS["button"],
            fg=self.COLORS["text"],
            activebackground=self.COLORS["button_active"],
            activeforeground=self.COLORS["text"],
            font=("Bahnschrift SemiBold", 9),
            relief="flat",
            bd=1,
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            highlightcolor=self.COLORS["accent"],
            padx=8,
            pady=2,
            cursor="hand2",
        ).pack(side="right", padx=8, pady=4)

        body = tk.Frame(panel, bg=self.COLORS["panel"])
        body.pack(fill="both", expand=True, padx=8, pady=8)

        self.avatar_canvas = tk.Canvas(
            body,
            width=245,
            height=260,
            bg="#070707",
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
            relief="flat",
        )
        self.avatar_canvas.pack(side="left", padx=(0, 8), pady=(0, 4))
        self._draw_avatar()

        right = tk.Frame(body, bg=self.COLORS["panel"])
        right.pack(side="left", fill="both", expand=True)

        self.ai_bubble_var = tk.StringVar(value="Boot sequence complete.")
        bubble = tk.Label(
            right,
            textvariable=self.ai_bubble_var,
            bg=self.COLORS["panel_alt"],
            fg="#E8E8E8",
            justify="left",
            anchor="nw",
            wraplength=402,
            font=("Consolas", 10),
            padx=8,
            pady=8,
        )
        bubble.pack(fill="x", pady=(0, 6))

        self.ai_feed_text = scrolledtext.ScrolledText(
            right,
            height=5,
            bg=self.COLORS["panel"],
            fg="#B8B8B8",
            insertbackground=self.COLORS["text"],
            relief="flat",
            bd=0,
            font=("Consolas", 9),
            wrap="word",
        )
        self.ai_feed_text.pack(fill="both", expand=True)
        self.ai_feed_text.configure(state="disabled")

    def _draw_avatar(self) -> None:
        c = self.avatar_canvas
        c.delete("all")

        w = int(c.cget("width"))
        h = int(c.cget("height"))
        cx = w // 2

        self.avatar_use_reference_face = False
        self.avatar_face_bounds = None
        self.avatar_matrix_streams = []

        # Background and digital ambience.
        c.create_rectangle(0, 0, w, h, fill="#07090C", outline="#242C35")
        for x in range(0, w, 12):
            shade = "#0E141A" if (x // 12) % 2 == 0 else "#0A0F14"
            c.create_line(x, 0, x, h, fill=shade)
        for y in range(0, h, 20):
            c.create_line(0, y, w, y, fill="#0B1116")

        # Prefer the exact face from the user-provided reference image if available.
        self.avatar_face_photo = self._build_reference_face_photo(target_w=198, target_h=216)
        if self.avatar_face_photo is not None:
            self.avatar_use_reference_face = True
            face_w = self.avatar_face_photo.width()
            face_h = self.avatar_face_photo.height()
            face_x1 = cx - (face_w // 2)
            face_y1 = max(10, (h - face_h) // 2 - 4)
            face_x2 = face_x1 + face_w
            face_y2 = face_y1 + face_h
            self.avatar_face_bounds = (face_x1, face_y1, face_x2, face_y2)

            c.create_rectangle(
                face_x1 - 4,
                face_y1 - 4,
                face_x2 + 4,
                face_y2 + 4,
                outline="#2A3644",
                width=1,
            )
            c.create_image(cx, face_y1 + face_h // 2, image=self.avatar_face_photo, anchor="center")
            c.create_oval(face_x1 - 12, face_y1 - 10, face_x2 + 12, face_y2 + 10, outline="#1A2631")

            # Binary streams over the portrait to preserve the 1/0 style.
            for x in range(8, w - 6, 15):
                bits = "\n".join(random.choice("01") for _ in range(22))
                y = random.uniform(-h, h - 24)
                stream_id = c.create_text(
                    x,
                    y,
                    text=bits,
                    anchor="n",
                    fill="#DDE3EA",
                    font=("Consolas", 8),
                )
                self.avatar_matrix_streams.append(
                    {"id": stream_id, "speed": random.uniform(1.2, 2.6), "x": x}
                )

            # Expression overlay anchors (kept subtle so the reference face remains visible).
            left_eye_x = face_x1 + int(face_w * 0.39)
            right_eye_x = face_x1 + int(face_w * 0.61)
            eye_y = face_y1 + int(face_h * 0.43)
            mouth_y = face_y1 + int(face_h * 0.71)

            self.avatar_eye_left = c.create_line(
                left_eye_x - 11, eye_y, left_eye_x + 11, eye_y, fill="#D6DDE5", width=2, smooth=True
            )
            self.avatar_eye_right = c.create_line(
                right_eye_x - 11, eye_y, right_eye_x + 11, eye_y, fill="#D6DDE5", width=2, smooth=True
            )
            self.avatar_pupil_left = c.create_oval(
                left_eye_x - 3, eye_y - 2, left_eye_x + 3, eye_y + 2, fill="#8FA7BC", outline=""
            )
            self.avatar_pupil_right = c.create_oval(
                right_eye_x - 3, eye_y - 2, right_eye_x + 3, eye_y + 2, fill="#8FA7BC", outline=""
            )
            self.avatar_brow_left = c.create_line(
                left_eye_x - 13, eye_y - 12, left_eye_x + 13, eye_y - 10, fill="#8A97A6", width=2, smooth=True
            )
            self.avatar_brow_right = c.create_line(
                right_eye_x - 13, eye_y - 10, right_eye_x + 13, eye_y - 12, fill="#8A97A6", width=2, smooth=True
            )
            self.avatar_mouth = c.create_arc(
                cx - 18,
                mouth_y - 9,
                cx + 18,
                mouth_y + 11,
                start=200,
                extent=140,
                style="arc",
                outline="#9BA7B4",
                width=2,
            )
            self.avatar_blush_left = c.create_oval(
                left_eye_x - 15,
                eye_y + 16,
                left_eye_x - 1,
                eye_y + 30,
                fill="#D08BA0",
                outline="",
                stipple="gray50",
                state="hidden",
            )
            self.avatar_blush_right = c.create_oval(
                right_eye_x + 1,
                eye_y + 16,
                right_eye_x + 15,
                eye_y + 30,
                fill="#D08BA0",
                outline="",
                stipple="gray50",
                state="hidden",
            )
            self.avatar_tear_left = c.create_oval(
                left_eye_x - 4,
                eye_y + 6,
                left_eye_x + 2,
                eye_y + 26,
                fill="#A6DBFF",
                outline="",
                state="hidden",
            )
            self.avatar_tear_right = c.create_oval(
                right_eye_x - 2,
                eye_y + 6,
                right_eye_x + 4,
                eye_y + 26,
                fill="#A6DBFF",
                outline="",
                state="hidden",
            )
            self.avatar_status_label = c.create_text(
                cx,
                h - 14,
                text="Monitoring",
                fill=self.COLORS["muted"],
                font=("Consolas", 10, "bold"),
            )
            self._render_avatar_face(self.avatar_expression, blink=False)
            return

        # Fallback vector face when no reference image is present.
        c.create_oval(cx - 88, 18, cx + 88, h - 10, outline="#1C2A35", width=1)
        c.create_oval(cx - 74, 26, cx + 74, h - 24, outline="#121B22", width=1)

        self.avatar_body = c.create_polygon(
            cx - 78,
            h - 2,
            cx - 55,
            192,
            cx - 22,
            172,
            cx + 22,
            172,
            cx + 55,
            192,
            cx + 78,
            h - 2,
            fill="#121923",
            outline="#2A384A",
            width=1,
        )
        self.avatar_neck = c.create_rectangle(
            cx - 16, 152, cx + 16, 178, fill="#E4EAF0", outline="#B6C2CE", width=1
        )
        self.avatar_choker = c.create_rectangle(
            cx - 20, 156, cx + 20, 164, fill="#0E151D", outline="#3E4E5F", width=1
        )

        self.avatar_hair_back = c.create_polygon(
            cx - 98,
            62,
            cx - 88,
            198,
            cx - 40,
            h - 2,
            cx + 40,
            h - 2,
            cx + 88,
            198,
            cx + 98,
            62,
            cx + 50,
            24,
            cx - 50,
            24,
            fill="#BFC8D2",
            outline="#657382",
            width=1,
        )
        self.avatar_face = c.create_oval(
            cx - 58, 58, cx + 58, 166, fill="#EAEFF4", outline="#AEBBC9", width=1
        )
        self.avatar_hair_front_left = c.create_polygon(
            cx - 82,
            66,
            cx - 72,
            190,
            cx - 22,
            162,
            cx - 28,
            86,
            fill="#B3BDC8",
            outline="#5E6C7B",
            width=1,
        )
        self.avatar_hair_front_right = c.create_polygon(
            cx + 82,
            66,
            cx + 72,
            190,
            cx + 22,
            162,
            cx + 28,
            86,
            fill="#B3BDC8",
            outline="#5E6C7B",
            width=1,
        )
        self.avatar_bang_center = c.create_polygon(
            cx - 30,
            52,
            cx - 6,
            126,
            cx + 6,
            126,
            cx + 30,
            52,
            fill="#AAB5C1",
            outline="#5C6A79",
            width=1,
        )

        self.avatar_eye_left = c.create_oval(
            cx - 52, 106, cx - 18, 120, fill="#F9FCFF", outline="#38424F", width=1
        )
        self.avatar_eye_right = c.create_oval(
            cx + 18, 106, cx + 52, 120, fill="#F9FCFF", outline="#38424F", width=1
        )
        self.avatar_pupil_left = c.create_oval(
            cx - 39, 109, cx - 30, 118, fill="#212A34", outline=""
        )
        self.avatar_pupil_right = c.create_oval(
            cx + 30, 109, cx + 39, 118, fill="#212A34", outline=""
        )
        self.avatar_brow_left = c.create_line(
            cx - 56, 99, cx - 16, 101, fill="#4F5D6D", width=2, smooth=True
        )
        self.avatar_brow_right = c.create_line(
            cx + 16, 101, cx + 56, 99, fill="#4F5D6D", width=2, smooth=True
        )
        self.avatar_nose = c.create_line(
            cx, 124, cx - 2, 132, fill="#AAB6C3", width=1, smooth=True
        )
        self.avatar_mouth = c.create_arc(
            cx - 20,
            136,
            cx + 20,
            162,
            start=200,
            extent=140,
            style="arc",
            outline="#4A5765",
            width=3,
        )
        self.avatar_blush_left = c.create_oval(
            cx - 66,
            126,
            cx - 50,
            141,
            fill="#D08BA0",
            outline="",
            stipple="gray50",
            state="hidden",
        )
        self.avatar_blush_right = c.create_oval(
            cx + 50,
            126,
            cx + 66,
            141,
            fill="#D08BA0",
            outline="",
            stipple="gray50",
            state="hidden",
        )
        self.avatar_tear_left = c.create_oval(
            cx - 35,
            122,
            cx - 29,
            142,
            fill="#A6DBFF",
            outline="",
            state="hidden",
        )
        self.avatar_tear_right = c.create_oval(
            cx + 29,
            122,
            cx + 35,
            142,
            fill="#A6DBFF",
            outline="",
            state="hidden",
        )
        self.avatar_status_label = c.create_text(
            cx,
            h - 14,
            text="Monitoring",
            fill=self.COLORS["muted"],
            font=("Consolas", 10, "bold"),
        )
        self._render_avatar_face(self.avatar_expression, blink=False)

    def _build_reference_face_photo(self, target_w: int, target_h: int) -> Optional[tk.PhotoImage]:
        candidate_paths = [
            self.avatar_reference_target,
            Path("assets") / "avatar_reference.png",
            Path("binary_anime_reference.png"),
            Path("avatar_reference.png"),
            Path("reference_avatar.png"),
        ]

        selected: Optional[Path] = None
        for path in candidate_paths:
            if path.exists():
                selected = path
                break
        if selected is None:
            self.avatar_reference_image = None
            self.avatar_reference_source = ""
            return None

        try:
            image = tk.PhotoImage(file=str(selected))
        except Exception:
            self.avatar_reference_image = None
            self.avatar_reference_source = ""
            return None

        self.avatar_reference_image = image
        self.avatar_reference_source = str(selected)

        src_w = max(1, image.width())
        src_h = max(1, image.height())

        x1 = int(src_w * 0.20)
        x2 = int(src_w * 0.80)
        y1 = int(src_h * 0.06)
        y2 = int(src_h * 0.63)
        if x2 <= x1 or y2 <= y1:
            return image

        crop = tk.PhotoImage()
        crop.tk.call(crop, "copy", image, "-from", x1, y1, x2, y2)
        scaled = crop

        down = max(1, int(max(scaled.width() / max(1, target_w), scaled.height() / max(1, target_h))))
        if down > 1:
            reduced = tk.PhotoImage()
            reduced.tk.call(reduced, "copy", scaled, "-subsample", down, down)
            scaled = reduced

        up = int(min(target_w / max(1, scaled.width()), target_h / max(1, scaled.height())))
        if up > 1:
            enlarged = tk.PhotoImage()
            enlarged.tk.call(enlarged, "copy", scaled, "-zoom", up, up)
            scaled = enlarged

        return scaled

    def _set_avatar_expression(self, expression: str) -> None:
        valid = {"neutral", "happy", "excited", "sad", "cry"}
        if expression not in valid:
            expression = "neutral"
        self.avatar_target_expression = expression

    def _update_avatar_emotion_from_metrics(self, metrics: Dict[str, Any]) -> None:
        total_trades = int(metrics.get("total_trades", 0))
        total_pnl = float(metrics.get("total_pnl", 0.0))
        daily_pnl = float(metrics.get("daily_pnl", 0.0))
        win_rate = float(metrics.get("win_rate", 0.0)) * 100.0

        delta_trades = total_trades - self.last_total_trades_for_face
        delta_pnl = total_pnl - self.last_total_pnl_for_face

        if total_trades <= 0:
            expression = "neutral"
        elif delta_trades > 0:
            if delta_pnl > 0.10:
                expression = "excited" if delta_pnl >= 2.0 else "happy"
            elif delta_pnl < -0.10:
                expression = "cry" if delta_pnl <= -1.5 else "sad"
            else:
                expression = "neutral"
        else:
            if total_pnl >= 3.0 or (daily_pnl >= 1.0 and win_rate >= 55.0):
                expression = "happy"
            elif total_pnl <= -1.5 or daily_pnl <= -0.8:
                expression = "sad"
            else:
                expression = "neutral"

        self._set_avatar_expression(expression)
        self.last_total_trades_for_face = total_trades
        self.last_total_pnl_for_face = total_pnl

    def _render_avatar_face(self, expression: str, blink: bool) -> None:
        if not hasattr(self, "avatar_eye_left"):
            return

        c = self.avatar_canvas
        w = int(c.cget("width"))
        h = int(c.cget("height"))
        cx = w // 2

        pupil_dx = int(2 * math.sin(self.avatar_tick * 0.11))
        pupil_dy = 0
        if expression in {"sad", "cry"}:
            pupil_dy = 1
        elif expression == "excited":
            pupil_dy = -1

        if self.avatar_use_reference_face and self.avatar_face_bounds is not None:
            x1, y1, x2, y2 = self.avatar_face_bounds
            fw = max(1, x2 - x1)
            fh = max(1, y2 - y1)

            left_eye_x = x1 + int(fw * 0.39)
            right_eye_x = x1 + int(fw * 0.61)
            eye_y = y1 + int(fh * 0.43)
            brow_left_x1 = left_eye_x - int(fw * 0.06)
            brow_left_x2 = left_eye_x + int(fw * 0.06)
            brow_right_x1 = right_eye_x - int(fw * 0.06)
            brow_right_x2 = right_eye_x + int(fw * 0.06)
            brow_y = eye_y - int(fh * 0.06)
            mouth_y = y1 + int(fh * 0.71)

            c.coords(self.avatar_eye_left, left_eye_x - 11, eye_y, left_eye_x + 11, eye_y)
            c.coords(self.avatar_eye_right, right_eye_x - 11, eye_y, right_eye_x + 11, eye_y)
            c.itemconfigure(self.avatar_eye_left, fill="#D6DDE5")
            c.itemconfigure(self.avatar_eye_right, fill="#D6DDE5")

            c.coords(
                self.avatar_pupil_left,
                left_eye_x - 3 + pupil_dx,
                eye_y - 2 + pupil_dy,
                left_eye_x + 3 + pupil_dx,
                eye_y + 2 + pupil_dy,
            )
            c.coords(
                self.avatar_pupil_right,
                right_eye_x - 3 + pupil_dx,
                eye_y - 2 + pupil_dy,
                right_eye_x + 3 + pupil_dx,
                eye_y + 2 + pupil_dy,
            )
            c.itemconfigure(self.avatar_pupil_left, state="hidden" if blink else "normal")
            c.itemconfigure(self.avatar_pupil_right, state="hidden" if blink else "normal")

            if expression == "happy":
                c.coords(self.avatar_brow_left, brow_left_x1, brow_y + 2, brow_left_x2, brow_y)
                c.coords(self.avatar_brow_right, brow_right_x1, brow_y, brow_right_x2, brow_y + 2)
                c.coords(self.avatar_mouth, cx - 20, mouth_y - 10, cx + 20, mouth_y + 10)
                c.itemconfigure(self.avatar_mouth, start=200, extent=140, outline="#76D1BD")
                c.itemconfigure(self.avatar_blush_left, state="normal")
                c.itemconfigure(self.avatar_blush_right, state="normal")
                c.itemconfigure(self.avatar_tear_left, state="hidden")
                c.itemconfigure(self.avatar_tear_right, state="hidden")
                c.itemconfigure(self.avatar_status_label, text="Profitable", fill=self.COLORS["success"])
            elif expression == "excited":
                c.coords(self.avatar_brow_left, brow_left_x1, brow_y + 1, brow_left_x2, brow_y - 2)
                c.coords(self.avatar_brow_right, brow_right_x1, brow_y - 2, brow_right_x2, brow_y + 1)
                c.coords(self.avatar_mouth, cx - 22, mouth_y - 12, cx + 22, mouth_y + 12)
                c.itemconfigure(self.avatar_mouth, start=195, extent=150, outline="#24CFA0")
                c.itemconfigure(self.avatar_blush_left, state="normal")
                c.itemconfigure(self.avatar_blush_right, state="normal")
                c.itemconfigure(self.avatar_tear_left, state="hidden")
                c.itemconfigure(self.avatar_tear_right, state="hidden")
                c.itemconfigure(self.avatar_status_label, text="Winning Streak", fill=self.COLORS["success"])
            elif expression == "sad":
                c.coords(self.avatar_brow_left, brow_left_x1, brow_y - 1, brow_left_x2, brow_y + 3)
                c.coords(self.avatar_brow_right, brow_right_x1, brow_y + 3, brow_right_x2, brow_y - 1)
                c.coords(self.avatar_mouth, cx - 18, mouth_y - 2, cx + 18, mouth_y + 12)
                c.itemconfigure(self.avatar_mouth, start=20, extent=140, outline="#A2AFBC")
                c.itemconfigure(self.avatar_blush_left, state="hidden")
                c.itemconfigure(self.avatar_blush_right, state="hidden")
                c.itemconfigure(self.avatar_tear_left, state="hidden")
                c.itemconfigure(self.avatar_tear_right, state="hidden")
                c.itemconfigure(self.avatar_status_label, text="Drawdown", fill=self.COLORS["warn"])
            elif expression == "cry":
                tear_shift = int((self.avatar_tick % 8) * 1.1)
                c.coords(self.avatar_brow_left, brow_left_x1, brow_y - 2, brow_left_x2, brow_y + 4)
                c.coords(self.avatar_brow_right, brow_right_x1, brow_y + 4, brow_right_x2, brow_y - 2)
                c.coords(self.avatar_mouth, cx - 18, mouth_y, cx + 18, mouth_y + 12)
                c.itemconfigure(self.avatar_mouth, start=20, extent=140, outline="#B0BCCC")
                c.itemconfigure(self.avatar_blush_left, state="hidden")
                c.itemconfigure(self.avatar_blush_right, state="hidden")
                c.coords(self.avatar_tear_left, left_eye_x - 4, eye_y + 6 + tear_shift, left_eye_x + 2, eye_y + 26 + tear_shift)
                c.coords(self.avatar_tear_right, right_eye_x - 2, eye_y + 6 + tear_shift, right_eye_x + 4, eye_y + 26 + tear_shift)
                c.itemconfigure(self.avatar_tear_left, state="normal")
                c.itemconfigure(self.avatar_tear_right, state="normal")
                c.itemconfigure(self.avatar_status_label, text="Losing Trade", fill=self.COLORS["danger"])
            else:
                c.coords(self.avatar_brow_left, brow_left_x1, brow_y + 1, brow_left_x2, brow_y + 1)
                c.coords(self.avatar_brow_right, brow_right_x1, brow_y + 1, brow_right_x2, brow_y + 1)
                c.coords(self.avatar_mouth, cx - 16, mouth_y - 4, cx + 16, mouth_y + 7)
                c.itemconfigure(self.avatar_mouth, start=200, extent=140, outline="#9BA7B4")
                c.itemconfigure(self.avatar_blush_left, state="hidden")
                c.itemconfigure(self.avatar_blush_right, state="hidden")
                c.itemconfigure(self.avatar_tear_left, state="hidden")
                c.itemconfigure(self.avatar_tear_right, state="hidden")
                c.itemconfigure(self.avatar_status_label, text="Monitoring", fill=self.COLORS["muted"])

            c.coords(self.avatar_status_label, cx, h - 14)
            for item in (
                self.avatar_brow_left,
                self.avatar_brow_right,
                self.avatar_eye_left,
                self.avatar_eye_right,
                self.avatar_pupil_left,
                self.avatar_pupil_right,
                self.avatar_mouth,
                self.avatar_blush_left,
                self.avatar_blush_right,
                self.avatar_tear_left,
                self.avatar_tear_right,
                self.avatar_status_label,
            ):
                c.tag_raise(item)
            return

        eye_y = 113
        eye_half_h = 7
        if blink:
            eye_half_h = 1
        elif expression == "excited":
            eye_half_h = 8
        elif expression in {"sad", "cry"}:
            eye_half_h = 6

        c.coords(self.avatar_eye_left, cx - 52, eye_y - eye_half_h, cx - 18, eye_y + eye_half_h)
        c.coords(self.avatar_eye_right, cx + 18, eye_y - eye_half_h, cx + 52, eye_y + eye_half_h)
        c.itemconfigure(self.avatar_eye_left, fill="#F9FCFF" if not blink else "#DEE6EE")
        c.itemconfigure(self.avatar_eye_right, fill="#F9FCFF" if not blink else "#DEE6EE")

        c.coords(
            self.avatar_pupil_left,
            cx - 39 + pupil_dx,
            eye_y - 4 + pupil_dy,
            cx - 30 + pupil_dx,
            eye_y + 5 + pupil_dy,
        )
        c.coords(
            self.avatar_pupil_right,
            cx + 30 + pupil_dx,
            eye_y - 4 + pupil_dy,
            cx + 39 + pupil_dx,
            eye_y + 5 + pupil_dy,
        )
        c.itemconfigure(self.avatar_pupil_left, state="hidden" if blink else "normal")
        c.itemconfigure(self.avatar_pupil_right, state="hidden" if blink else "normal")

        if expression == "happy":
            c.coords(self.avatar_brow_left, cx - 56, 100, cx - 16, 98)
            c.coords(self.avatar_brow_right, cx + 16, 98, cx + 56, 100)
            c.coords(self.avatar_mouth, cx - 21, 136, cx + 21, 164)
            c.itemconfigure(self.avatar_mouth, start=200, extent=140, outline="#40596A")
            c.itemconfigure(self.avatar_blush_left, state="normal")
            c.itemconfigure(self.avatar_blush_right, state="normal")
            c.itemconfigure(self.avatar_tear_left, state="hidden")
            c.itemconfigure(self.avatar_tear_right, state="hidden")
            c.itemconfigure(self.avatar_status_label, text="Profitable", fill=self.COLORS["success"])
        elif expression == "excited":
            c.coords(self.avatar_brow_left, cx - 56, 97, cx - 16, 94)
            c.coords(self.avatar_brow_right, cx + 16, 94, cx + 56, 97)
            c.coords(self.avatar_mouth, cx - 22, 132, cx + 22, 168)
            c.itemconfigure(self.avatar_mouth, start=195, extent=150, outline="#24CFA0")
            c.itemconfigure(self.avatar_blush_left, state="normal")
            c.itemconfigure(self.avatar_blush_right, state="normal")
            c.itemconfigure(self.avatar_tear_left, state="hidden")
            c.itemconfigure(self.avatar_tear_right, state="hidden")
            c.itemconfigure(self.avatar_status_label, text="Winning Streak", fill=self.COLORS["success"])
        elif expression == "sad":
            c.coords(self.avatar_brow_left, cx - 56, 96, cx - 16, 102)
            c.coords(self.avatar_brow_right, cx + 16, 102, cx + 56, 96)
            c.coords(self.avatar_mouth, cx - 18, 144, cx + 18, 166)
            c.itemconfigure(self.avatar_mouth, start=20, extent=140, outline="#6A7A8B")
            c.itemconfigure(self.avatar_blush_left, state="hidden")
            c.itemconfigure(self.avatar_blush_right, state="hidden")
            c.itemconfigure(self.avatar_tear_left, state="hidden")
            c.itemconfigure(self.avatar_tear_right, state="hidden")
            c.itemconfigure(self.avatar_status_label, text="Drawdown", fill=self.COLORS["warn"])
        elif expression == "cry":
            tear_shift = int((self.avatar_tick % 8) * 1.2)
            c.coords(self.avatar_brow_left, cx - 56, 95, cx - 16, 104)
            c.coords(self.avatar_brow_right, cx + 16, 104, cx + 56, 95)
            c.coords(self.avatar_mouth, cx - 18, 146, cx + 18, 167)
            c.itemconfigure(self.avatar_mouth, start=20, extent=140, outline="#8293A5")
            c.itemconfigure(self.avatar_blush_left, state="hidden")
            c.itemconfigure(self.avatar_blush_right, state="hidden")
            c.coords(self.avatar_tear_left, cx - 35, 122 + tear_shift, cx - 29, 142 + tear_shift)
            c.coords(self.avatar_tear_right, cx + 29, 122 + tear_shift, cx + 35, 142 + tear_shift)
            c.itemconfigure(self.avatar_tear_left, state="normal")
            c.itemconfigure(self.avatar_tear_right, state="normal")
            c.itemconfigure(self.avatar_status_label, text="Losing Trade", fill=self.COLORS["danger"])
        else:
            c.coords(self.avatar_brow_left, cx - 56, 99, cx - 16, 100)
            c.coords(self.avatar_brow_right, cx + 16, 100, cx + 56, 99)
            c.coords(self.avatar_mouth, cx - 18, 142, cx + 18, 161)
            c.itemconfigure(self.avatar_mouth, start=200, extent=140, outline="#617180")
            c.itemconfigure(self.avatar_blush_left, state="hidden")
            c.itemconfigure(self.avatar_blush_right, state="hidden")
            c.itemconfigure(self.avatar_tear_left, state="hidden")
            c.itemconfigure(self.avatar_tear_right, state="hidden")
            c.itemconfigure(self.avatar_status_label, text="Monitoring", fill=self.COLORS["muted"])

    def _gray_hex(self, level: int) -> str:
        level = max(0, min(255, int(level)))
        return f"#{level:02X}{level:02X}{level:02X}"

    @staticmethod
    def _pixel_to_rgb(pixel: Any) -> Optional[tuple[int, int, int]]:
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            return int(pixel[0]), int(pixel[1]), int(pixel[2])
        if isinstance(pixel, str):
            p = pixel.strip()
            if p.startswith("#") and len(p) == 7:
                try:
                    return int(p[1:3], 16), int(p[3:5], 16), int(p[5:7], 16)
                except ValueError:
                    return None
        return None

    def _load_reference_tone_map(
        self, cols: int, rows: int
    ) -> Optional[Dict[str, Any]]:
        candidate_paths = [
            self.avatar_reference_target,
            Path("assets") / "avatar_reference.png",
            Path("binary_anime_reference.png"),
            Path("avatar_reference.png"),
            Path("reference_avatar.png"),
        ]

        selected: Optional[Path] = None
        for path in candidate_paths:
            if path.exists():
                selected = path
                break

        if selected is None:
            self.avatar_reference_image = None
            self.avatar_reference_source = ""
            return None

        try:
            image = tk.PhotoImage(file=str(selected))
        except Exception:
            self.avatar_reference_image = None
            self.avatar_reference_source = ""
            return None

        self.avatar_reference_image = image
        self.avatar_reference_source = str(selected)

        src_w = max(1, image.width())
        src_h = max(1, image.height())
        raw_lum: List[List[float]] = []
        lum_min = 255.0
        lum_max = 0.0

        for row in range(rows):
            sy = min(src_h - 1, int(row * src_h / rows))
            lum_row: List[float] = []
            for col in range(cols):
                sx = min(src_w - 1, int(col * src_w / cols))
                rgb = self._pixel_to_rgb(image.get(sx, sy))
                if rgb is None:
                    lum = 0.0
                else:
                    r, g, b = rgb
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                lum_row.append(lum)
                lum_min = min(lum_min, lum)
                lum_max = max(lum_max, lum)
            raw_lum.append(lum_row)

        lum_span = max(1.0, lum_max - lum_min)
        tone_map: List[List[Optional[int]]] = []
        edge_map: List[List[float]] = []

        for row in range(rows):
            tone_row: List[Optional[int]] = []
            edge_row: List[float] = []
            for col in range(cols):
                lum = raw_lum[row][col]
                left = raw_lum[row][max(0, col - 1)]
                right = raw_lum[row][min(cols - 1, col + 1)]
                up = raw_lum[max(0, row - 1)][col]
                down = raw_lum[min(rows - 1, row + 1)][col]

                gx = right - left
                gy = down - up
                edge = min(1.0, (abs(gx) + abs(gy)) / 175.0)
                edge_row.append(edge)

                norm = (lum - lum_min) / lum_span
                # Higher contrast for line-art detail retention.
                contrast = (norm - 0.5) * 1.8 + 0.5
                contrast = max(0.0, min(1.0, contrast))
                tone = int((contrast ** 0.92) * 255.0)
                # Preserve outlines and facial edges.
                tone = int(tone - edge * 95.0)

                if lum <= 2.0:
                    tone_row.append(None)
                    continue
                tone_row.append(max(16, min(252, tone)))
            tone_map.append(tone_row)
            edge_map.append(edge_row)

        return {
            "tones": tone_map,
            "edges": edge_map,
            "source": str(selected),
        }

    def load_reference_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Binary Reference Image",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.avatar_reference_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_path, self.avatar_reference_target)
            self.avatar_tick = 0
            self._draw_avatar()
            self._queue_ai_message(
                f"Reference face loaded: {self.avatar_reference_target.name}. "
                "Avatar now uses the exact reference face with binary overlay."
            )
            self.logger.info(f"Avatar reference image loaded: {file_path}")
        except Exception as exc:
            self.logger.error(f"Failed to load avatar reference image: {exc}")
            messagebox.showerror("Reference Image", f"Failed to load image.\n{exc}")

    @staticmethod
    def _in_ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float) -> bool:
        if rx <= 0.0 or ry <= 0.0:
            return False
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    def _compute_portrait_tone(self, nx: float, ny: float) -> Optional[int]:
        # Global silhouette components.
        hair_main = self._in_ellipse(nx, ny, 0.50, 0.27, 0.235, 0.22)
        hair_left = self._in_ellipse(nx, ny, 0.30, 0.63, 0.255, 0.45)
        hair_right = self._in_ellipse(nx, ny, 0.70, 0.63, 0.255, 0.45)
        hair_mid = 0.23 <= nx <= 0.77 and 0.19 <= ny <= 0.92
        hair = (hair_main or hair_left or hair_right or hair_mid) and ny <= 0.98

        face_oval = self._in_ellipse(nx, ny, 0.50, 0.39, 0.165, 0.205)
        chin_ok = ny <= 0.50 or abs(nx - 0.50) <= (0.165 - (ny - 0.50) * 0.48)
        face = face_oval and chin_ok and ny >= 0.20
        neck = 0.462 <= nx <= 0.538 and 0.54 <= ny <= 0.63
        shoulders = self._in_ellipse(nx, ny, 0.50, 0.74, 0.36, 0.18)
        torso = 0.62 <= ny <= 0.99 and abs(nx - 0.50) <= (0.33 - 0.22 * (ny - 0.62))
        body = shoulders or torso

        # Remove hard outside areas.
        if not (hair or face or neck or body):
            return None

        # Base grayscale tones with directional lighting.
        tone: Optional[int] = None
        if hair:
            hair_light = 1.0 - min(1.0, abs(nx - 0.50) / 0.30)
            tone = int(148 + 62 * hair_light - 34 * max(0.0, ny - 0.42))
        if body and tone is None:
            body_light = 1.0 - min(1.0, abs(nx - 0.50) / 0.34)
            tone = int(118 + 56 * body_light - 34 * max(0.0, ny - 0.66))
        if face:
            face_light = 1.0 - min(1.0, abs(nx - 0.50) / 0.17)
            tone = int(175 + 52 * face_light - 32 * max(0.0, ny - 0.41))
        if neck:
            neck_light = 1.0 - min(1.0, abs(nx - 0.50) / 0.08)
            tone = int(146 + 44 * neck_light)

        # Hair strands and bangs (anime framing).
        if hair and 0.19 <= ny <= 0.54:
            strand_centers = (0.34, 0.40, 0.46, 0.52, 0.58, 0.64)
            for index, center in enumerate(strand_centers):
                width = 0.018 + 0.003 * (index % 2)
                spread = width + 0.085 * max(0.0, ny - 0.20)
                if abs(nx - center) <= spread and ny >= (0.20 + 0.10 * abs(center - 0.50)):
                    tone = 142 + index * 6
            if abs(nx - 0.50) < 0.012 and 0.20 <= ny <= 0.49:
                tone = 128

        # Face details.
        if face and 0.34 <= ny <= 0.48 and abs(nx - 0.50) < 0.10:
            # Eye whites.
            if self._in_ellipse(nx, ny, 0.435, 0.395, 0.058, 0.024) or self._in_ellipse(
                nx, ny, 0.565, 0.395, 0.058, 0.024
            ):
                tone = 238
            # Iris and pupil.
            if self._in_ellipse(nx, ny, 0.435, 0.397, 0.021, 0.018) or self._in_ellipse(
                nx, ny, 0.565, 0.397, 0.021, 0.018
            ):
                tone = 70
            if self._in_ellipse(nx, ny, 0.435, 0.398, 0.011, 0.010) or self._in_ellipse(
                nx, ny, 0.565, 0.398, 0.011, 0.010
            ):
                tone = 38
            # Eye highlights.
            if self._in_ellipse(nx, ny, 0.428, 0.391, 0.005, 0.004) or self._in_ellipse(
                nx, ny, 0.558, 0.391, 0.005, 0.004
            ):
                tone = 252
            # Lashes and lids.
            if abs(ny - (0.375 + 0.11 * abs(nx - 0.435))) <= 0.003 and abs(nx - 0.435) <= 0.062:
                tone = 58
            if abs(ny - (0.375 + 0.11 * abs(nx - 0.565))) <= 0.003 and abs(nx - 0.565) <= 0.062:
                tone = 58
            # Brows.
            if abs(ny - 0.356) <= 0.004 and abs(nx - 0.435) <= 0.060:
                tone = 78
            if abs(ny - 0.356) <= 0.004 and abs(nx - 0.565) <= 0.060:
                tone = 78

        # Nose and mouth.
        if face and abs(nx - 0.50) <= 0.012 and 0.438 <= ny <= 0.456:
            tone = 124
        if face and 0.470 <= nx <= 0.530 and abs(ny - 0.488) <= 0.006:
            tone = 84
        if face and 0.470 <= nx <= 0.530 and abs(ny - 0.501) <= 0.004:
            tone = 132

        # Choker and outfit edges.
        if 0.425 <= nx <= 0.575 and 0.565 <= ny <= 0.603:
            tone = 70
        if body and 0.64 <= ny <= 0.70 and abs(nx - 0.50) <= 0.30:
            tone = min(170, int((tone or 130) - 22))
        if body and ny >= 0.71 and abs(nx - 0.50) <= 0.052:
            tone = 90

        # Hair edge highlights and shoulder glints.
        if hair and tone is not None:
            edge_glint = (
                (abs(nx - 0.20) < 0.014 and ny >= 0.28)
                or (abs(nx - 0.80) < 0.014 and ny >= 0.28)
                or (0.26 <= ny <= 0.38 and abs(abs(nx - 0.50) - 0.16) < 0.010)
            )
            if edge_glint:
                tone = 236
        if body and 0.69 <= ny <= 0.76 and (abs(nx - 0.35) < 0.010 or abs(nx - 0.65) < 0.010):
            tone = 214

        # Fine binary texture to avoid flat regions.
        if tone is not None:
            grain = math.sin(nx * 82.0 + ny * 22.0) * 0.5 + math.sin(ny * 96.0) * 0.5
            tone = int(tone + 7 * grain)
            tone = max(22, min(252, tone))

        return tone

    def _make_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=self.COLORS["panel"],
            highlightthickness=1,
            highlightbackground=self.COLORS["border"],
        )
        title_label = tk.Label(
            card,
            text=title,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Bahnschrift SemiBold", 16),
        )
        title_label.pack(anchor="w", padx=12, pady=(10, 6))

        body = tk.Frame(card, bg=self.COLORS["panel"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        card.body = body  # type: ignore[attr-defined]
        return card

    def _create_entry_row(
        self, card: tk.Frame, label: str, variable: tk.StringVar, masked: bool = False
    ) -> None:
        row = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        row.pack(fill="x", pady=4)
        tk.Label(
            row,
            text=label,
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            width=24,
            anchor="w",
            font=("Bahnschrift SemiBold", 11),
        ).pack(side="left")
        tk.Entry(
            row,
            textvariable=variable,
            show="*" if masked else "",
            bg=self.COLORS["panel_alt"],
            fg=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            font=("Consolas", 11),
            width=40,
        ).pack(side="left", fill="x", expand=True)

    def _create_status_block(self, card: tk.Frame, key: str, title: str) -> None:
        row = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        row.pack(fill="x", pady=8)

        tk.Label(
            row,
            text=title,
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Bahnschrift SemiBold", 14),
            anchor="w",
        ).pack(anchor="w")

        bar_canvas = tk.Canvas(
            row,
            width=280,
            height=20,
            bg=self.COLORS["panel"],
            highlightthickness=0,
            bd=0,
        )
        bar_canvas.pack(anchor="w", pady=(4, 3))
        segments: List[int] = []
        x = 0
        for _ in range(20):
            seg = bar_canvas.create_rectangle(
                x,
                3,
                x + 12,
                16,
                fill=self.COLORS["glow_low"],
                outline=self.COLORS["panel"],
            )
            segments.append(seg)
            x += 14

        state_label = tk.Label(
            row,
            text="Offline",
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Consolas", 10),
            anchor="w",
        )
        state_label.pack(anchor="w")

        self.status_widgets[key] = {
            "canvas": bar_canvas,
            "segments": segments,
            "state_label": state_label,
        }

    def _create_label_value(self, card: tk.Frame, label: str, key: str) -> None:
        row = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        row.pack(fill="x", pady=2)
        tk.Label(
            row,
            text=label,
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Consolas", 11),
            anchor="w",
            width=14,
        ).pack(side="left")
        value = tk.Label(
            row,
            text="N/A",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Consolas", 11, "bold"),
            anchor="e",
        )
        value.pack(side="right")
        self.account_value_labels[key] = value

    def _create_perf_value(self, card: tk.Frame, label: str, key: str) -> None:
        row = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        row.pack(fill="x", pady=2)
        tk.Label(
            row,
            text=label,
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Consolas", 11),
            anchor="w",
            width=14,
        ).pack(side="left")
        value = tk.Label(
            row,
            text="0",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Consolas", 11, "bold"),
            anchor="e",
        )
        value.pack(side="right")
        self.perf_value_labels[key] = value

    def _create_risk_value(self, card: tk.Frame, label: str, key: str) -> None:
        row = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        row.pack(fill="x", pady=2)
        tk.Label(
            row,
            text=label,
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Consolas", 11),
            anchor="w",
            width=16,
        ).pack(side="left")
        value = tk.Label(
            row,
            text="0",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Consolas", 11, "bold"),
            anchor="e",
        )
        value.pack(side="right")
        self.risk_value_labels[key] = value

    def _create_summary_value(
        self, card: tk.Frame, title: str, key: str, column: int
    ) -> None:
        frame = tk.Frame(card.body, bg=self.COLORS["panel"])  # type: ignore[attr-defined]
        frame.grid(row=0, column=column, padx=6, pady=6, sticky="ew")
        tk.Label(
            frame,
            text=title,
            bg=self.COLORS["panel"],
            fg=self.COLORS["muted"],
            font=("Bahnschrift SemiBold", 12),
        ).pack(anchor="center")
        value = tk.Label(
            frame,
            text="0",
            bg=self.COLORS["panel"],
            fg=self.COLORS["text"],
            font=("Bahnschrift SemiBold", 22),
        )
        value.pack(anchor="center", pady=(4, 0))
        self.summary_value_labels[key] = value

    def _show_page(self, page_id: str) -> None:
        if page_id not in self.pages:
            return

        self.current_page = page_id
        self.pages[page_id].tkraise()

        for key, button in self.nav_buttons.items():
            if key == page_id:
                button.configure(
                    bg=self.COLORS["accent"],
                    fg="#00130F",
                    activebackground="#00D6A9",
                    activeforeground="#00130F",
                )
            else:
                button.configure(
                    bg=self.COLORS["button"],
                    fg=self.COLORS["text"],
                    activebackground=self.COLORS["button_active"],
                    activeforeground=self.COLORS["text"],
                )

    def _initialize_log_offset(self) -> None:
        if self.log_file_path.exists():
            self.log_file_offset = self.log_file_path.stat().st_size

    def _schedule_refresh(self) -> None:
        self._refresh_ui()
        interval = max(1.0, self._safe_float(self.update_interval_var.get(), 5.0))
        self.root.after(int(interval * 1000), self._schedule_refresh)

    def _refresh_ui(self) -> None:
        try:
            self._refresh_status()
            self._refresh_account()
            self._refresh_model_panel()
            self._refresh_trading_data()
            self._refresh_risk_monitoring()
            self._refresh_log_stream()
        except Exception as exc:
            self.logger.error(f"UI refresh error: {exc}")
    def _refresh_status(self) -> None:
        mt5_ok = self.mt5_connector.is_connected()
        model_ok = self.model_manager.is_model_loaded()
        trading_ok = bool(
            self.is_trading
            and self.trading_engine is not None
            and getattr(self.trading_engine, "is_running", False)
        )

        self._set_status_row("mt5", mt5_ok, "Connected" if mt5_ok else "Disconnected")
        self._set_status_row("model", model_ok, "Loaded" if model_ok else "Not Loaded")
        self._set_status_row("trading", trading_ok, "Running" if trading_ok else "Stopped")

    def _set_status_row(self, key: str, ok: bool, state_text: str) -> None:
        widget = self.status_widgets.get(key)
        if not widget:
            return

        segments = widget["segments"]
        canvas = widget["canvas"]
        state_label = widget["state_label"]

        active_count = len(segments) if ok else 2
        active_color = self.COLORS["success"] if ok else self.COLORS["warn"]
        for index, seg in enumerate(segments):
            color = active_color if index < active_count else self.COLORS["glow_low"]
            canvas.itemconfigure(seg, fill=color)

        state_label.configure(
            text=state_text,
            fg=self.COLORS["success"] if ok else self.COLORS["muted"],
        )

    def _refresh_account(self) -> None:
        if not self.mt5_connector.is_connected():
            for label in self.account_value_labels.values():
                label.configure(text="N/A")
            return

        account = self.mt5_connector.get_account_info() or {}
        self.account_value_labels["account"].configure(text=str(account.get("login", "N/A")))
        self.account_value_labels["balance"].configure(
            text=f"${float(account.get('balance', 0.0)):.2f}"
        )
        self.account_value_labels["equity"].configure(
            text=f"${float(account.get('equity', 0.0)):.2f}"
        )
        self.account_value_labels["margin"].configure(
            text=f"${float(account.get('margin', 0.0)):.2f}"
        )
        self.account_value_labels["free_margin"].configure(
            text=f"${float(account.get('margin_free', 0.0)):.2f}"
        )

    def _refresh_model_panel(self) -> None:
        if self.model_manager.is_model_loaded():
            model_info = self.model_manager.get_model_info()
            self.model_status_text.delete("1.0", tk.END)
            self.model_status_text.insert(
                "1.0",
                "Model status: LOADED\n"
                f"Feature dimensions: {model_info.get('feature_dimension', 'N/A')}\n"
                f"Parameters: {model_info.get('total_parameters', 'N/A')}\n"
                f"Estimated model size: {model_info.get('model_size_mb', 0):.4f} MB\n",
            )
            self.model_info_text.delete("1.0", tk.END)
            self.model_info_text.insert("1.0", json.dumps(model_info, indent=2))
        else:
            self.model_status_text.delete("1.0", tk.END)
            self.model_status_text.insert("1.0", "Model status: NOT LOADED\n")

    def _refresh_trading_data(self) -> None:
        if self.trading_engine is None:
            self._update_metrics({})
            self._update_signal_views([])
            self._update_position_views([])
            return

        metrics = self.trading_engine.get_performance_metrics() or {}
        signals = self.trading_engine.get_active_signals() or []
        positions = self.trading_engine.get_active_positions() or []

        self.latest_metrics = metrics
        self.latest_signals = signals
        self.latest_positions = positions

        self._update_metrics(metrics)
        self._update_signal_views(signals)
        self._update_position_views(positions)
        self._emit_ai_snapshot(metrics, signals, positions)

    def _refresh_risk_monitoring(self) -> None:
        """Refresh risk monitoring display"""
        try:
            if self.trading_engine is None or not hasattr(self.trading_engine, 'tail_risk_protector'):
                # Set default values when trading engine not available
                default_values = {
                    "current_drawdown": "0.0%",
                    "volatility_regime": "NORMAL",
                    "risk_level": "MODERATE",
                    "tail_risk_score": "0.0",
                    "dynamic_confidence": "0.78"
                }
                for key, value in default_values.items():
                    if key in self.risk_value_labels:
                        self.risk_value_labels[key].configure(text=value)
                return
            
            # Get risk report from tail risk protector
            risk_report = self.trading_engine.tail_risk_protector.get_risk_report()
            
            if 'error' in risk_report:
                self.logger.warning(f"Risk monitoring error: {risk_report['error']}")
                return
            
            # Update risk values
            current_state = risk_report.get('current_risk_state', {})
            
            # Current drawdown
            drawdown = current_state.get('drawdown', 0.0)
            self.risk_value_labels["current_drawdown"].configure(
                text=f"{drawdown:.1%}",
                fg=self.COLORS["danger"] if drawdown > 0.05 else self.COLORS["text"]
            )
            
            # Volatility regime
            vol_regime = current_state.get('volatility_regime', 'NORMAL_VOLATILITY')
            vol_color = {
                'LOW_VOLATILITY': self.COLORS["success"],
                'NORMAL_VOLATILITY': self.COLORS["text"],
                'HIGH_VOLATILITY': self.COLORS["warn"],
                'EXTREME_VOLATILITY': self.COLORS["danger"]
            }.get(vol_regime, self.COLORS["text"])
            
            self.risk_value_labels["volatility_regime"].configure(
                text=vol_regime.replace('_', ' ').title(),
                fg=vol_color
            )
            
            # Risk level
            risk_level = current_state.get('risk_level', 'MODERATE')
            risk_color = {
                'MINIMAL': self.COLORS["success"],
                'LOW': self.COLORS["success"],
                'MODERATE': self.COLORS["warn"],
                'HIGH': self.COLORS["danger"],
                'CRITICAL': self.COLORS["danger"]
            }.get(risk_level, self.COLORS["text"])
            
            self.risk_value_labels["risk_level"].configure(
                text=risk_level,
                fg=risk_color
            )
            
            # Tail risk score (based on recent performance)
            recent_perf = risk_report.get('recent_performance', {})
            win_rate = recent_perf.get('win_rate', 0.5)
            tail_risk_score = max(0, (0.5 - win_rate) * 2)  # Higher score = more risk
            
            self.risk_value_labels["tail_risk_score"].configure(
                text=f"{tail_risk_score:.2f}",
                fg=self.COLORS["success"] if tail_risk_score < 0.3 else 
                   self.COLORS["warn"] if tail_risk_score < 0.6 else self.COLORS["danger"]
            )
            
            # Dynamic confidence (from latest adjustment)
            confidence_adjustments = risk_report.get('confidence_adjustments', [])
            if confidence_adjustments:
                latest_adjustment = confidence_adjustments[-1]
                adjusted_conf = latest_adjustment.get('adjusted_confidence', 0.78)
                self.risk_value_labels["dynamic_confidence"].configure(
                    text=f"{adjusted_conf:.3f}",
                    fg=self.COLORS["success"] if adjusted_conf > 0.8 else self.COLORS["text"]
                )
            
        except Exception as e:
            self.logger.error(f"Error refreshing risk monitoring: {e}")

    def _update_metrics(self, metrics: Dict[str, Any]) -> None:
        win_rate = float(metrics.get("win_rate", 0.0)) * 100.0
        total_trades = int(metrics.get("total_trades", 0))
        winning_trades = int(metrics.get("winning_trades", 0))
        daily_pnl = float(metrics.get("daily_pnl", 0.0))
        total_pnl = float(metrics.get("total_pnl", 0.0))

        self.perf_value_labels["win_rate"].configure(text=f"{win_rate:.1f}%")
        self.perf_value_labels["total_trades"].configure(text=str(total_trades))
        self.perf_value_labels["winning_trades"].configure(text=str(winning_trades))
        self.perf_value_labels["daily_pnl"].configure(text=f"${daily_pnl:.2f}")
        self.perf_value_labels["total_pnl"].configure(text=f"${total_pnl:.2f}")

        pnl_color = self.COLORS["success"] if total_pnl >= 0 else self.COLORS["danger"]
        self.perf_value_labels["total_pnl"].configure(fg=pnl_color)
        self.perf_value_labels["daily_pnl"].configure(
            fg=self.COLORS["success"] if daily_pnl >= 0 else self.COLORS["danger"]
        )

        self.summary_value_labels["sum_win_rate"].configure(text=f"{win_rate:.1f}%")
        self.summary_value_labels["sum_total_trades"].configure(text=str(total_trades))
        self.summary_value_labels["sum_total_pnl"].configure(
            text=f"${total_pnl:.2f}", fg=pnl_color
        )
        self._update_avatar_emotion_from_metrics(metrics)

    def _update_signal_views(self, signals: List[Dict[str, Any]]) -> None:
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)

        recent = list(reversed(signals[-30:]))
        for signal in recent:
            timestamp = self._format_time(signal.get("timestamp"))
            symbol = signal.get("symbol", "N/A")
            action = signal.get("action", "N/A")
            confidence = f"{float(signal.get('confidence', 0.0)) * 100.0:.1f}%"
            entry = f"{float(signal.get('entry_price', 0.0)):.5f}"
            reason = signal.get("reason", "")
            self.signals_tree.insert(
                "",
                tk.END,
                values=(timestamp, symbol, action, confidence, entry, reason),
            )

        self.dashboard_signal_text.delete("1.0", tk.END)
        if not recent:
            self.dashboard_signal_text.insert("1.0", "No active signals yet.")
        else:
            lines = []
            for signal in recent[:14]:
                ts = self._format_time(signal.get("timestamp"))
                symbol = signal.get("symbol", "N/A")
                action = signal.get("action", "N/A")
                conf = float(signal.get("confidence", 0.0)) * 100.0
                price = float(signal.get("entry_price", 0.0))
                lines.append(f"{ts}  {symbol}  {action:<4}  {conf:5.1f}%  @ {price:.5f}")
            self.dashboard_signal_text.insert("1.0", "\n".join(lines))

    def _update_position_views(self, positions: List[Dict[str, Any]]) -> None:
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)

        for position in positions:
            ticket = position.get("ticket", "N/A")
            symbol = position.get("symbol", "N/A")
            side = position.get("action", "N/A")
            entry = f"{float(position.get('entry_price', 0.0)):.5f}"
            current = f"{float(position.get('current_price', 0.0)):.5f}"
            pnl = float(position.get("unrealized_pnl", 0.0))
            self.positions_tree.insert(
                "",
                tk.END,
                values=(ticket, symbol, side, entry, current, f"${pnl:.2f}"),
            )

        self.dashboard_positions_text.delete("1.0", tk.END)
        if not positions:
            self.dashboard_positions_text.insert("1.0", "No open positions.")
        else:
            lines = []
            for position in positions[:14]:
                symbol = position.get("symbol", "N/A")
                side = position.get("action", "N/A")
                entry = float(position.get("entry_price", 0.0))
                current = float(position.get("current_price", 0.0))
                pnl = float(position.get("unrealized_pnl", 0.0))
                lines.append(
                    f"{symbol}  {side:<4}  entry {entry:.5f} -> now {current:.5f}  pnl ${pnl:.2f}"
                )
            self.dashboard_positions_text.insert("1.0", "\n".join(lines))

    def _refresh_log_stream(self) -> None:
        if not self.log_file_path.exists():
            return

        try:
            with open(self.log_file_path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.log_file_offset)
                chunk = handle.read()
                self.log_file_offset = handle.tell()
        except Exception as exc:
            self.logger.error(f"Failed to read incremental logs: {exc}")
            return

        if not chunk:
            return

        if self.current_page == "logs":
            self.log_text.insert(tk.END, chunk)
            self.log_text.see(tk.END)

        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        for line in lines[-6:]:
            if "ERROR" in line:
                self._queue_ai_message("Execution error detected. Inspect Logs tab now.")
            elif "Trade executed" in line:
                self._queue_ai_message("Order confirmation received from engine.")

    def _emit_ai_snapshot(
        self,
        metrics: Dict[str, Any],
        signals: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
    ) -> None:
        now = datetime.now()
        if (now - self.last_ai_emit_ts).total_seconds() < 12:
            return

        if not signals:
            msg = "No fresh signals in queue. Monitoring M5 bars for the next setup."
            key = "no-signals"
        else:
            latest = signals[-1]
            action = str(latest.get("action", "HOLD"))
            symbol = str(latest.get("symbol", "N/A"))
            confidence = float(latest.get("confidence", 0.0)) * 100.0
            open_count = len(positions)
            total_pnl = float(metrics.get("total_pnl", 0.0))
            msg = (
                f"{symbol} signal is {action} at {confidence:.1f}% confidence. "
                f"Open positions: {open_count}. Net P&L: ${total_pnl:.2f}."
            )
            key = f"{symbol}:{action}:{int(confidence)}:{open_count}:{int(total_pnl)}"

        if key != self.last_ai_emit_key:
            self.last_ai_emit_key = key
            self.last_ai_emit_ts = now
            self._queue_ai_message(msg)

    def _queue_ai_message(self, message: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, lambda: self._queue_ai_message(message))
            return

        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.ai_messages.append(line)
        self.ai_bubble_var.set(message)
        self._append_ai_feed(line)

    def _append_ai_feed(self, line: str) -> None:
        self.ai_feed_text.configure(state="normal")
        self.ai_feed_text.insert(tk.END, line + "\n")
        self.ai_feed_text.see(tk.END)
        self.ai_feed_text.configure(state="disabled")

    def _animate_avatar(self) -> None:
        self.avatar_tick += 1

        if not hasattr(self, "avatar_eye_left"):
            self.root.after(160, self._animate_avatar)
            return

        if self.avatar_use_reference_face and self.avatar_matrix_streams:
            h = int(self.avatar_canvas.cget("height"))
            for stream in self.avatar_matrix_streams:
                item_id = stream.get("id")
                x = float(stream.get("x", 0.0))
                speed = float(stream.get("speed", 1.0))
                pos = self.avatar_canvas.coords(item_id)
                if not pos:
                    continue
                y = float(pos[1]) + speed
                if y > h + 25:
                    y = random.uniform(-h, -15)
                    bits = "\n".join(random.choice("01") for _ in range(22))
                    self.avatar_canvas.itemconfigure(item_id, text=bits)
                elif self.avatar_tick % 7 == 0 and random.random() < 0.35:
                    bits = "\n".join(random.choice("01") for _ in range(22))
                    self.avatar_canvas.itemconfigure(item_id, text=bits)
                self.avatar_canvas.coords(item_id, x, y)

        if self.avatar_expression != self.avatar_target_expression and self.avatar_tick % 3 == 0:
            self.avatar_expression = self.avatar_target_expression

        blink = self.avatar_tick % 26 in (0, 1)
        self._render_avatar_face(self.avatar_expression, blink)
        self.root.after(120, self._animate_avatar)

    def connect_mt5(self) -> None:
        self.connect_btn.configure(state="disabled", text="Connecting...")
        self.bottom_status.configure(text="Connecting to MT5...")

        server = self.server_var.get().strip()
        login = self.login_var.get().strip()
        password = self.password_var.get().strip()
        use_manual = bool(server and login and password and server.lower() != "auto")

        def worker() -> None:
            try:
                success = (
                    self.mt5_connector.connect(server=server, login=login, password=password)
                    if use_manual
                    else self.mt5_connector.connect()
                )
                self.root.after(0, lambda: self._on_connect_done(success))
            except Exception as exc:
                self.root.after(0, lambda: self._on_connect_done(False, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_connect_done(self, success: bool, error: str = "") -> None:
        self.connect_btn.configure(state="normal", text="Connect MT5")
        if success:
            self.mt5_connected = True
            self.bottom_status.configure(text="MT5 connected")
            self.logger.info("MT5 connection established")
            self._queue_ai_message("MT5 link established. Market telemetry is now live.")
            messagebox.showinfo("MT5", "Connected to MT5 successfully.")
        else:
            self.mt5_connected = False
            self.bottom_status.configure(text="MT5 connection failed")
            self.logger.error(f"MT5 connection failed: {error}")
            self._queue_ai_message("MT5 connection failed. Check account, server, and terminal.")
            messagebox.showerror("MT5", f"Failed to connect to MT5.\n{error}")

    def load_model(self) -> None:
        self.load_model_btn.configure(state="disabled", text="Loading...")
        self.bottom_status.configure(text="Loading neural model...")

        def worker() -> None:
            try:
                success = self.model_manager.load_model()
                self.root.after(0, lambda: self._on_model_loaded(success))
            except Exception as exc:
                self.root.after(0, lambda: self._on_model_loaded(False, str(exc)))

        threading.Thread(target=worker, daemon=True).start()
    def _on_model_loaded(self, success: bool, error: str = "") -> None:
        self.load_model_btn.configure(state="normal", text="Load Model")
        if success:
            self.model_loaded = True
            self.bottom_status.configure(text="Model loaded")
            self.logger.info("Neural model loaded")
            self._queue_ai_message("Model weights loaded. Signal engine is ready.")
            self._refresh_model_panel()
            messagebox.showinfo("Model", "Neural model loaded successfully.")
        else:
            self.model_loaded = False
            self.bottom_status.configure(text="Model load failed")
            self.logger.error(f"Model loading failed: {error}")
            self._queue_ai_message("Model load failed. Verify neural_model.pth integrity.")
            messagebox.showerror("Model", f"Failed to load model.\n{error}")

    def start_trading(self) -> None:
        if not self.mt5_connector.is_connected():
            messagebox.showwarning("Trading", "Connect MT5 before starting trading.")
            return
        if not self.model_manager.is_model_loaded():
            messagebox.showwarning("Trading", "Load a model before starting trading.")
            return

        risk = max(0.1, self._safe_float(self.risk_var.get(), 1.5)) / 100.0
        confidence = max(50.0, self._safe_float(self.confidence_var.get(), 65.0)) / 100.0
        max_positions = max(1, self._safe_int(self.max_trades_var.get(), 5))
        selected_pairs = [pair for pair, var in self.pair_vars.items() if var.get()]
        if not selected_pairs:
            messagebox.showwarning("Trading", "Select at least one symbol.")
            return

        try:
            self.trading_engine = TradingEngine(
                mt5_connector=self.mt5_connector,
                model_manager=self.model_manager,
                risk_per_trade=risk,
                confidence_threshold=confidence,
                trading_pairs=selected_pairs,
                max_concurrent_positions=max_positions,
            )
            # Immediate execution profile requested by user:
            # trade quickly on valid signals across selected symbols.
            self.trading_engine.immediate_trade_mode = True
            self.trading_engine.profitability_first_mode = False
            self.trading_engine.model_pattern_conflict_block = False
            self.trading_engine.mtf_alignment_enabled = False
            self.trading_engine.tail_risk_control_enabled = False
            self.trading_engine.model_min_trade_score = min(
                self.trading_engine.model_min_trade_score, 0.33
            )
            self.trading_engine.model_min_directional_gap = min(
                self.trading_engine.model_min_directional_gap, 0.005
            )
            self.trading_engine.symbol_entry_cooldown_seconds = min(
                self.trading_engine.symbol_entry_cooldown_seconds, 60
            )
            self.trading_engine.max_new_trades_per_hour = max(
                self.trading_engine.max_new_trades_per_hour, len(selected_pairs) * 2
            )
            self.trading_engine.start()
            self.is_trading = True
            self.start_trading_btn.configure(state="disabled")
            self.stop_trading_btn.configure(state="normal")
            self.bottom_status.configure(text="Trading engine running")
            self.logger.info("Trading engine started")
            self._queue_ai_message(
                f"Trading activated on {len(selected_pairs)} symbols. Monitoring executions now."
            )
            messagebox.showinfo("Trading", "Trading engine started.")
        except Exception as exc:
            self.logger.error(f"Trading startup error: {exc}")
            self._queue_ai_message("Trading engine start failed. Review configuration and logs.")
            messagebox.showerror("Trading", f"Failed to start trading.\n{exc}")

    def stop_trading(self) -> None:
        try:
            if self.trading_engine is not None:
                self.trading_engine.stop()
            self.is_trading = False
            self.start_trading_btn.configure(state="normal")
            self.stop_trading_btn.configure(state="disabled")
            self.bottom_status.configure(text="Trading engine stopped")
            self.logger.info("Trading engine stopped")
            self._queue_ai_message("Trading halted. Positions remain under broker control.")
            messagebox.showinfo("Trading", "Trading engine stopped.")
        except Exception as exc:
            self.logger.error(f"Trading shutdown error: {exc}")
            messagebox.showerror("Trading", f"Failed to stop trading.\n{exc}")

    def train_model(self) -> None:
        self.bottom_status.configure(text="Training model (this may take a few minutes)...")
        self._queue_ai_message("Launching symbol-aware training pipeline with walk-forward validation.")

        trainer_script = Path(__file__).resolve().parent / "simple_neural_trainer.py"
        if not trainer_script.exists():
            self.bottom_status.configure(text="Training script not found")
            messagebox.showerror("Model", f"Training script not found:\n{trainer_script}")
            return

        def worker() -> None:
            try:
                result = subprocess.run(
                    [sys.executable, str(trainer_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).resolve().parent),
                )
                self.root.after(
                    0,
                    lambda: self._on_train_model_done(
                        success=(result.returncode == 0),
                        stdout=result.stdout,
                        stderr=result.stderr,
                    ),
                )
            except Exception as exc:
                self.root.after(0, lambda: self._on_train_model_done(False, "", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_train_model_done(self, success: bool, stdout: str, stderr: str) -> None:
        tail_lines = 20
        out_tail = "\n".join((stdout or "").splitlines()[-tail_lines:])
        err_tail = "\n".join((stderr or "").splitlines()[-tail_lines:])

        if success:
            self.bottom_status.configure(text="Model training completed")
            self._queue_ai_message("Training completed. Reloading updated neural model.")
            reload_success = self.model_manager.load_model()
            self.model_loaded = bool(reload_success)
            self._refresh_model_panel()

            if reload_success:
                messagebox.showinfo(
                    "Model",
                    "Training completed and model reloaded successfully.\n\n"
                    f"Trainer output:\n{out_tail}"
                )
            else:
                messagebox.showwarning(
                    "Model",
                    "Training completed, but model reload failed.\n"
                    "Use Load Model and check logs.\n\n"
                    f"Trainer output:\n{out_tail}"
                )
            return

        self.bottom_status.configure(text="Model training failed")
        self._queue_ai_message("Training failed. Review trainer logs for details.")
        detail = err_tail or out_tail or "No additional output."
        messagebox.showerror(
            "Model",
            "Model training failed.\n\n"
            f"Details:\n{detail}"
        )

    def validate_model(self) -> None:
        info = self.model_manager.get_model_info()
        if info.get("error"):
            messagebox.showwarning("Model", "No loaded model to validate.")
            return
        self._queue_ai_message("Model metadata checked. Use backtests before live deployment.")
        messagebox.showinfo("Model", "Model appears loaded and structurally valid.")

    def clear_logs(self) -> None:
        self.log_text.delete("1.0", tk.END)

    def export_logs(self) -> None:
        if not self.log_file_path.exists():
            messagebox.showwarning("Logs", "No log file available to export.")
            return

        export_path = filedialog.asksaveasfilename(
            title="Export Logs",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"trading_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        if not export_path:
            return

        try:
            shutil.copy2(self.log_file_path, export_path)
            messagebox.showinfo("Logs", f"Logs exported to:\n{export_path}")
        except Exception as exc:
            messagebox.showerror("Logs", f"Failed to export logs.\n{exc}")

    def refresh_logs(self) -> None:
        if not self.log_file_path.exists():
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert("1.0", "No log file available yet.")
            return

        content = self._read_tail_lines(self.log_file_path, line_count=1200)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", content)
        self.log_text.see(tk.END)

    def _read_tail_lines(self, path: Path, line_count: int = 600) -> str:
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                file_size = handle.tell()
                block_size = 4096
                blocks = []
                lines_found = 0
                cursor = file_size

                while cursor > 0 and lines_found <= line_count:
                    read_size = min(block_size, cursor)
                    cursor -= read_size
                    handle.seek(cursor)
                    block = handle.read(read_size)
                    blocks.append(block)
                    lines_found += block.count(b"\n")

                data = b"".join(reversed(blocks))
            text = data.decode("utf-8", errors="replace")
            split_lines = text.splitlines()
            return "\n".join(split_lines[-line_count:])
        except Exception as exc:
            self.logger.error(f"Failed to read log tail: {exc}")
            return f"Failed to read logs: {exc}"

    def save_settings(self) -> None:
        try:
            server = self.server_var.get().strip() or "auto"
            login = self.login_var.get().strip() or "auto"
            password = self.password_var.get().strip() or "auto"
            interval = max(1, self._safe_int(self.update_interval_var.get(), 5))
            max_positions = max(1, self._safe_int(self.max_trades_var.get(), 5))
            risk_pct = max(0.1, self._safe_float(self.risk_var.get(), 1.5))
            conf_pct = min(95.0, max(50.0, self._safe_float(self.confidence_var.get(), 65.0)))

            self.config_manager.set_config("user", "mt5.server", server)
            self.config_manager.set_config("user", "mt5.login", login)
            self.config_manager.set_config("user", "mt5.password", password)
            self.config_manager.set_config("main", "update_interval", interval)
            self.config_manager.set_config(
                "trading", "general.max_concurrent_positions", max_positions
            )
            self.config_manager.set_config(
                "trading", "general.default_risk_per_trade", risk_pct
            )
            self.config_manager.set_config(
                "trading", "general.default_confidence_threshold", conf_pct
            )

            for pair, var in self.pair_vars.items():
                if pair == "BTCUSD":
                    group = "crypto_pairs"
                elif pair in ("USDCAD", "NZDUSD", "EURJPY", "GBPJPY"):
                    group = "minor_pairs"
                else:
                    group = "major_pairs"
                self.config_manager.set_config(
                    "trading",
                    f"trading_pairs.{group}.{pair}.enabled",
                    bool(var.get()),
                )

            self.config_manager.save_config("all")
            self._queue_ai_message("Settings persisted. New values are now active in the UI.")
            messagebox.showinfo("Settings", "Settings saved successfully.")
        except Exception as exc:
            messagebox.showerror("Settings", f"Failed to save settings.\n{exc}")

    def _load_saved_settings_into_ui(self) -> None:
        self.server_var.set(str(self.config_manager.get_config("user", "mt5.server", "auto")))
        login = self.config_manager.get_config("user", "mt5.login", "auto")
        password = self.config_manager.get_config("user", "mt5.password", "auto")
        self.login_var.set("" if login == "auto" else str(login))
        self.password_var.set("" if password == "auto" else str(password))

        self.update_interval_var.set(
            str(int(self.config_manager.get_config("main", "update_interval", 5)))
        )
        self.max_trades_var.set(
            str(
                int(
                    self.config_manager.get_config(
                        "trading", "general.max_concurrent_positions", 5
                    )
                )
            )
        )
        self.risk_var.set(
            str(
                float(
                    self.config_manager.get_config(
                        "trading", "general.default_risk_per_trade", 1.5
                    )
                )
            )
        )
        self.confidence_var.set(
            str(
                float(
                    self.config_manager.get_config(
                        "trading", "general.default_confidence_threshold", 65
                    )
                )
            )
        )

        for pair, var in self.pair_vars.items():
            if pair == "BTCUSD":
                key = f"trading_pairs.crypto_pairs.{pair}.enabled"
            elif pair in ("USDCAD", "NZDUSD", "EURJPY", "GBPJPY"):
                key = f"trading_pairs.minor_pairs.{pair}.enabled"
            else:
                key = f"trading_pairs.major_pairs.{pair}.enabled"
            enabled = self.config_manager.get_config("trading", key, True)
            var.set(bool(enabled))

    def _safe_float(self, value: str, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value: str, default: int) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    def _format_time(self, iso_or_value: Any) -> str:
        try:
            if isinstance(iso_or_value, str):
                dt = datetime.fromisoformat(iso_or_value.replace("Z", "+00:00"))
                return dt.strftime("%H:%M:%S")
            return str(iso_or_value)
        except Exception:
            return str(iso_or_value)

    def _on_close(self) -> None:
        try:
            if self.trading_engine is not None:
                self.trading_engine.stop()
            if self.mt5_connector.is_connected():
                self.mt5_connector.disconnect()
        except Exception as exc:
            self.logger.error(f"Shutdown error: {exc}")
        finally:
            self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
        NeuralTradingApp(root)
        root.mainloop()
    except Exception as exc:
        print(f"Application error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

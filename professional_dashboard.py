#!/usr/bin/env python3
"""
XEO - Neural Trading Dashboard
================================
Premium fintech dashboard with glassmorphism cards, gradient accents,
and Tailwind-inspired spacing. Dark theme with neon green accent.
"""

import sys
import os
import re
import logging
import threading
import queue
import tkinter as tk
from datetime import datetime, timedelta
from typing import Optional, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from app.mt5_connector import MT5Connector
from app.model_manager import NeuralModelManager
from app.trading_engine import TradingEngine
from agentic_orchestrator import AgenticOrchestrator

ALL_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "NZDUSD", "EURJPY", "GBPJPY", "BTCUSD",
]
MODEL_PATH = "neural_model.pth"

# ── XEO Design Tokens (PayTech-inspired) ────────────────────────
BG_PRIMARY    = "#060612"    # Near-black with blue tint
BG_SECONDARY  = "#0C0C1D"   # Slightly lighter
BG_CARD       = "#12122A"   # Card background
BG_CARD_ALT   = "#16163A"   # Hover / alternate card
BG_SIDEBAR    = "#09091A"   # Sidebar
BG_INPUT      = "#1A1A3E"   # Input fields
ACCENT        = "#C8FF00"   # Neon green-yellow
ACCENT_DIM    = "#6B8700"   # Dimmed accent
ACCENT_GLOW   = "#C8FF0020" # Accent with alpha (for CSS-like glow)
GREEN         = "#00E676"
GREEN_DIM     = "#00E67630"
RED           = "#FF3D5A"
RED_DIM       = "#FF3D5A30"
YELLOW        = "#FFD600"
BLUE          = "#5C7CFA"
CYAN          = "#18FFFF"
TEXT_WHITE    = "#F0F0F5"
TEXT_GRAY     = "#9090A8"
TEXT_MUTED    = "#505068"
BORDER        = "#1E1E3A"
BORDER_GLOW   = "#2A2A50"
CARD_BORDER   = "#252548"

# Spacing scale (Tailwind-inspired: 1=4px)
SP1, SP2, SP3, SP4, SP5, SP6, SP8 = 4, 8, 12, 16, 20, 24, 32


class QueueLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue: queue.Queue = queue.Queue(maxsize=5000)

    def emit(self, record):
        try:
            self.log_queue.put_nowait((record.levelno, self.format(record)))
        except queue.Full:
            pass


class RoundedFrame(tk.Canvas):
    """A canvas that draws a rounded rectangle background, simulating CSS border-radius."""
    def __init__(self, parent, bg=BG_CARD, border_color=CARD_BORDER,
                 radius=16, border_width=1, **kwargs):
        super().__init__(parent, highlightthickness=0, bg=parent["bg"], **kwargs)
        self._bg_color = bg
        self._border_color = border_color
        self._radius = radius
        self._border_width = border_width
        self._inner_frame = tk.Frame(self, bg=bg)
        self._inner_window = self.create_window(0, 0, anchor="nw", window=self._inner_frame)
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event=None):
        self.delete("bg_rect")
        w, h = self.winfo_width(), self.winfo_height()
        r = self._radius
        bw = self._border_width
        # Outer border
        if bw > 0:
            self._rounded_rect(bw//2, bw//2, w - bw//2, h - bw//2, r,
                               fill=self._bg_color, outline=self._border_color,
                               width=bw, tags="bg_rect")
        else:
            self._rounded_rect(0, 0, w, h, r, fill=self._bg_color,
                               outline="", tags="bg_rect")
        self.tag_lower("bg_rect")
        # Resize inner frame
        pad = self._radius // 3
        self.coords(self._inner_window, pad, pad)
        self._inner_frame.configure(width=w - pad * 2, height=h - pad * 2)

    def _rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    @property
    def inner(self):
        return self._inner_frame


class XeoDashboard(tk.Tk):
    """Premium fintech trading dashboard."""

    def __init__(self):
        super().__init__()

        self.title("XEO")
        self.geometry("1500x900")
        self.minsize(1200, 750)
        self.configure(bg=BG_PRIMARY)

        # State
        self.mt5: Optional[MT5Connector] = None
        self.model_mgr: Optional[NeuralModelManager] = None
        self.engine: Optional[TradingEngine] = None
        self.orchestrator = None
        self.journal = None
        self._is_live = False
        self._stop_event = threading.Event()
        self._current_page = "dashboard"
        self._log_lines: list = []

        # Logging
        self.log_handler = QueueLogHandler()
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s"))
        logging.getLogger().addHandler(self.log_handler)

        # Fonts
        self.font_brand = ("Segoe UI", 26, "bold")
        self.font_brand_sub = ("Segoe UI", 10)
        self.font_nav = ("Segoe UI", 12)
        self.font_nav_active = ("Segoe UI", 12, "bold")
        self.font_h1 = ("Segoe UI", 32, "bold")
        self.font_h2 = ("Segoe UI", 18, "bold")
        self.font_h3 = ("Segoe UI", 14, "bold")
        self.font_body = ("Segoe UI", 12)
        self.font_small = ("Segoe UI", 10)
        self.font_tiny = ("Segoe UI", 9)
        self.font_mono = ("Consolas", 10)
        self.font_metric_value = ("Segoe UI", 22, "bold")
        self.font_metric_label = ("Segoe UI", 10)
        self.font_account_value = ("Segoe UI", 38, "bold")

        self._build_ui()

        self.after(500, self._connect_and_start)
        self.after(2000, self._tick_fast)
        self.after(15000, self._tick_slow)
        self.after(500, self._tick_logs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Sidebar
        self._build_sidebar()
        # Main content area
        self.main_area = tk.Frame(self, bg=BG_PRIMARY)
        self.main_area.pack(side="left", fill="both", expand=True)
        # Pages
        self.pages: Dict[str, tk.Frame] = {}
        self._build_page_dashboard()
        self._build_page_positions()
        self._build_page_performance()
        self._build_page_risk()
        self._build_page_settings()
        self._build_page_logs()
        self._show_page("dashboard")

    # ── SIDEBAR ──────────────────────────────────────────────────

    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, bg=BG_SIDEBAR, width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Brand
        brand = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        brand.pack(fill="x", padx=SP5, pady=(SP8, SP1))
        tk.Label(brand, text="XEO", font=self.font_brand, fg=ACCENT,
                 bg=BG_SIDEBAR, anchor="w").pack(anchor="w")
        tk.Label(brand, text="Neural Trading System", font=self.font_brand_sub,
                 fg=TEXT_MUTED, bg=BG_SIDEBAR, anchor="w").pack(anchor="w")

        # Accent line under brand
        accent_line = tk.Frame(self.sidebar, bg=ACCENT, height=2)
        accent_line.pack(fill="x", padx=SP5, pady=(SP3, SP4))

        # Navigation items
        self.nav_frames: Dict[str, tk.Frame] = {}
        self.nav_labels: Dict[str, tk.Label] = {}
        self.nav_indicators: Dict[str, tk.Frame] = {}

        nav_items = [
            ("dashboard",   "Dashboard"),
            ("positions",   "Positions"),
            ("performance", "Performance"),
            ("risk",        "Risk"),
            ("settings",    "Settings"),
            ("logs",        "Logs"),
        ]

        for key, label in nav_items:
            row = tk.Frame(self.sidebar, bg=BG_SIDEBAR, cursor="hand2")
            row.pack(fill="x", pady=1)

            # Active indicator bar (left edge)
            indicator = tk.Frame(row, bg=BG_SIDEBAR, width=3)
            indicator.pack(side="left", fill="y")

            lbl = tk.Label(row, text=f"  {label}", font=self.font_nav,
                           fg=TEXT_GRAY, bg=BG_SIDEBAR, anchor="w",
                           padx=SP4, pady=SP3)
            lbl.pack(fill="x")

            # Bind click
            for widget in (row, lbl):
                widget.bind("<Button-1>", lambda e, k=key: self._show_page(k))
                widget.bind("<Enter>", lambda e, r=row, l=lbl: self._nav_hover(r, l, True))
                widget.bind("<Leave>", lambda e, r=row, l=lbl, k=key: self._nav_hover(r, l, False, k))

            self.nav_frames[key] = row
            self.nav_labels[key] = lbl
            self.nav_indicators[key] = indicator

        # Bottom spacer
        spacer = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        spacer.pack(fill="both", expand=True)

        # Status indicator
        tk.Frame(self.sidebar, bg=BORDER, height=1).pack(fill="x", padx=SP4, pady=(0, SP2))

        self.status_frame = tk.Frame(self.sidebar, bg=BG_SIDEBAR)
        self.status_frame.pack(fill="x", padx=SP5, pady=(0, SP5))
        self.status_dot_canvas = tk.Canvas(self.status_frame, width=8, height=8,
                                            bg=BG_SIDEBAR, highlightthickness=0)
        self.status_dot_canvas.pack(side="left", padx=(0, SP2))
        self.status_dot_canvas.create_oval(1, 1, 7, 7, fill=YELLOW, outline="")
        self.status_label = tk.Label(self.status_frame, text="Connecting...",
                                      font=self.font_tiny, fg=YELLOW, bg=BG_SIDEBAR)
        self.status_label.pack(side="left")

        # Version
        tk.Label(self.sidebar, text="v1.0.0", font=self.font_tiny,
                 fg=TEXT_MUTED, bg=BG_SIDEBAR).pack(pady=(0, SP3))

    def _nav_hover(self, row, lbl, entering, key=None):
        if entering:
            row.configure(bg=BG_CARD)
            lbl.configure(bg=BG_CARD)
        else:
            is_active = key == self._current_page if key else False
            bg = BG_CARD_ALT if is_active else BG_SIDEBAR
            row.configure(bg=bg)
            lbl.configure(bg=bg)

    def _show_page(self, page_key: str):
        self._current_page = page_key
        for key, frame in self.pages.items():
            if key == page_key:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        # Update nav highlighting
        for key in self.nav_frames:
            is_active = key == page_key
            bg = BG_CARD_ALT if is_active else BG_SIDEBAR
            self.nav_frames[key].configure(bg=bg)
            self.nav_labels[key].configure(
                bg=bg,
                fg=ACCENT if is_active else TEXT_GRAY,
                font=self.font_nav_active if is_active else self.font_nav,
            )
            self.nav_indicators[key].configure(bg=ACCENT if is_active else BG_SIDEBAR)

    # ══════════════════════════════════════════════════════════════
    # PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════

    def _build_page_dashboard(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["dashboard"] = page

        # Top header bar
        header = tk.Frame(page, bg=BG_PRIMARY)
        header.pack(fill="x", padx=SP8, pady=(SP6, 0))

        # Left: Account info
        acct_frame = tk.Frame(header, bg=BG_PRIMARY)
        acct_frame.pack(side="left")
        tk.Label(acct_frame, text="Account Value", font=self.font_small,
                 fg=TEXT_GRAY, bg=BG_PRIMARY).pack(anchor="w")
        self.lbl_account_value = tk.Label(acct_frame, text="$0.00",
                                           font=self.font_account_value,
                                           fg=TEXT_WHITE, bg=BG_PRIMARY)
        self.lbl_account_value.pack(anchor="w")
        self.lbl_account_change = tk.Label(acct_frame, text="+0.00%",
                                            font=self.font_small, fg=GREEN,
                                            bg=BG_PRIMARY)
        self.lbl_account_change.pack(anchor="w")

        # Right: Status badge
        self.conn_badge = tk.Label(header, text="  CONNECTING  ",
                                    font=("Segoe UI", 9, "bold"),
                                    fg=BG_PRIMARY, bg=YELLOW)
        self.conn_badge.pack(side="right", pady=SP2)

        # ── Main content: 2-column layout ──
        content = tk.Frame(page, bg=BG_PRIMARY)
        content.pack(fill="both", expand=True, padx=SP8, pady=SP4)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # LEFT COLUMN: Equity Chart
        self.chart_outer = tk.Frame(content, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                     highlightthickness=1, relief="flat")
        self.chart_outer.grid(row=0, column=0, sticky="nsew", padx=(0, SP3), pady=0)

        chart_header = tk.Frame(self.chart_outer, bg=BG_CARD)
        chart_header.pack(fill="x", padx=SP5, pady=(SP4, 0))
        tk.Label(chart_header, text="Equity Curve", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(side="left")
        self.chart_badge = tk.Label(chart_header, text="LIVE", font=self.font_tiny,
                                     fg=GREEN, bg=BG_CARD)
        self.chart_badge.pack(side="right")

        self.chart_container = tk.Frame(self.chart_outer, bg=BG_CARD)
        self.chart_container.pack(fill="both", expand=True, padx=SP2, pady=SP2)
        self.chart_placeholder = tk.Label(self.chart_container,
                                           text="Waiting for trade data...",
                                           font=self.font_body, fg=TEXT_MUTED, bg=BG_CARD)
        self.chart_placeholder.pack(expand=True)
        self._equity_canvas = None

        # RIGHT COLUMN: Portfolio Risk + Recent Trades
        right = tk.Frame(content, bg=BG_PRIMARY)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)

        # Risk Score Card
        risk_card = tk.Frame(right, bg=BG_CARD, highlightbackground=CARD_BORDER,
                              highlightthickness=1)
        risk_card.grid(row=0, column=0, sticky="ew", pady=(0, SP3))
        rc_inner = tk.Frame(risk_card, bg=BG_CARD)
        rc_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(rc_inner, text="Portfolio Risk", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self._risk_label = tk.Label(rc_inner, text="Low Risk", font=self.font_small,
                                     fg=GREEN, bg=BG_CARD)
        self._risk_label.pack(anchor="w", pady=(SP1, SP2))
        # Progress bar (canvas-based)
        self._risk_bar_canvas = tk.Canvas(rc_inner, height=8, bg=BORDER,
                                           highlightthickness=0)
        self._risk_bar_canvas.pack(fill="x")
        self._risk_bar_fill = self._risk_bar_canvas.create_rectangle(
            0, 0, 30, 8, fill=GREEN, outline="")
        self._risk_bar_value = 0.15

        # Recent Trades Card
        trades_card = tk.Frame(right, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                highlightthickness=1)
        trades_card.grid(row=1, column=0, sticky="nsew")

        tc_header = tk.Frame(trades_card, bg=BG_CARD)
        tc_header.pack(fill="x", padx=SP5, pady=(SP4, SP2))
        tk.Label(tc_header, text="Recent Trades", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(side="left")
        self.trade_count_lbl = tk.Label(tc_header, text="0 open", font=self.font_tiny,
                                         fg=TEXT_MUTED, bg=BG_CARD)
        self.trade_count_lbl.pack(side="right")

        # Scrollable trades area
        self.trades_canvas = tk.Canvas(trades_card, bg=BG_CARD, highlightthickness=0)
        trades_scroll = tk.Scrollbar(trades_card, orient="vertical",
                                      command=self.trades_canvas.yview)
        self.trades_inner = tk.Frame(self.trades_canvas, bg=BG_CARD)
        self.trades_inner.bind("<Configure>",
                                lambda e: self.trades_canvas.configure(
                                    scrollregion=self.trades_canvas.bbox("all")))
        self.trades_canvas.create_window((0, 0), window=self.trades_inner, anchor="nw")
        self.trades_canvas.configure(yscrollcommand=trades_scroll.set)
        self.trades_canvas.pack(side="left", fill="both", expand=True, padx=SP2, pady=(0, SP2))
        trades_scroll.pack(side="right", fill="y")

        # ── Bottom Metric Cards ──
        bottom = tk.Frame(page, bg=BG_PRIMARY)
        bottom.pack(fill="x", padx=SP8, pady=(SP3, SP5))
        bottom.columnconfigure((0, 1, 2, 3), weight=1)

        self.card_realized = self._make_stat_card(bottom, "Realized P/L", "$0.00", GREEN, 0)
        self.card_unrealized = self._make_stat_card(bottom, "Unrealized P/L", "$0.00", BLUE, 1)
        self.card_daily = self._make_stat_card(bottom, "Daily P/L", "$0.00", CYAN, 2)
        self.card_drawdown = self._make_stat_card(bottom, "Max Drawdown", "0.00%", RED, 3)

    def _make_stat_card(self, parent, title, value, accent_color, col):
        """Create a bottom stat card with colored top accent line."""
        outer = tk.Frame(parent, bg=BG_CARD, highlightbackground=CARD_BORDER,
                          highlightthickness=1)
        outer.grid(row=0, column=col, sticky="nsew", padx=SP1, pady=0)

        # Top accent line
        tk.Frame(outer, bg=accent_color, height=3).pack(fill="x")

        inner = tk.Frame(outer, bg=BG_CARD)
        inner.pack(fill="x", padx=SP4, pady=SP3)

        tk.Label(inner, text=title, font=self.font_metric_label,
                 fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
        val_label = tk.Label(inner, text=value, font=self.font_metric_value,
                              fg=TEXT_WHITE, bg=BG_CARD)
        val_label.pack(anchor="w", pady=(SP1, 0))
        return val_label

    def _make_trade_card(self, parent, symbol, direction, pnl, lots, entry, current):
        """Create a single trade entry card with accent bar."""
        is_buy = direction == "BUY"
        accent = GREEN if is_buy else RED
        pnl_color = GREEN if pnl >= 0 else RED

        card = tk.Frame(parent, bg=BG_CARD_ALT, highlightbackground=BORDER,
                         highlightthickness=1)
        card.pack(fill="x", padx=SP2, pady=2)

        # Left accent bar
        tk.Frame(card, bg=accent, width=4).pack(side="left", fill="y")

        # Content
        body = tk.Frame(card, bg=BG_CARD_ALT)
        body.pack(fill="x", padx=SP3, pady=SP2)

        # Top row: symbol + direction | P/L
        top = tk.Frame(body, bg=BG_CARD_ALT)
        top.pack(fill="x")

        left_info = tk.Frame(top, bg=BG_CARD_ALT)
        left_info.pack(side="left")
        tk.Label(left_info, text=symbol, font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD_ALT).pack(side="left")
        dir_badge = tk.Label(left_info, text=f" {direction} ",
                              font=("Segoe UI", 8, "bold"),
                              fg=BG_PRIMARY, bg=accent)
        dir_badge.pack(side="left", padx=(SP2, 0))

        pnl_txt = f"${pnl:+,.2f}"
        tk.Label(top, text=pnl_txt, font=("Segoe UI", 14, "bold"),
                 fg=pnl_color, bg=BG_CARD_ALT).pack(side="right")

        # Bottom row: entry/current + lots
        bot = tk.Frame(body, bg=BG_CARD_ALT)
        bot.pack(fill="x", pady=(2, 0))
        tk.Label(bot, text=f"Entry {entry:.5f}  |  Now {current:.5f}",
                 font=self.font_tiny, fg=TEXT_MUTED, bg=BG_CARD_ALT).pack(side="left")
        tk.Label(bot, text=f"{lots:.2f} lots", font=self.font_tiny,
                 fg=TEXT_GRAY, bg=BG_CARD_ALT).pack(side="right")

    # ══════════════════════════════════════════════════════════════
    # PAGE: POSITIONS
    # ══════════════════════════════════════════════════════════════

    def _build_page_positions(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["positions"] = page

        header = tk.Frame(page, bg=BG_PRIMARY)
        header.pack(fill="x", padx=SP8, pady=(SP6, SP3))
        tk.Label(header, text="Open Positions", font=self.font_h2,
                 fg=TEXT_WHITE, bg=BG_PRIMARY).pack(side="left")
        self.pos_count_label = tk.Label(header, text="0 positions", font=self.font_small,
                                         fg=TEXT_MUTED, bg=BG_PRIMARY)
        self.pos_count_label.pack(side="right")

        # Table container
        table_frame = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                highlightthickness=1)
        table_frame.pack(fill="both", expand=True, padx=SP8, pady=(0, SP5))

        # Header row
        self.pos_header_frame = tk.Frame(table_frame, bg=BG_CARD_ALT)
        self.pos_header_frame.pack(fill="x")
        cols = ["Symbol", "Direction", "Entry", "Current", "SL", "TP", "Lots", "P/L"]
        for i, col in enumerate(cols):
            tk.Label(self.pos_header_frame, text=col, font=("Segoe UI", 10, "bold"),
                     fg=TEXT_MUTED, bg=BG_CARD_ALT, width=12, anchor="w").pack(
                side="left", padx=SP3, pady=SP2)

        # Scrollable body
        self.pos_canvas = tk.Canvas(table_frame, bg=BG_CARD, highlightthickness=0)
        pos_scroll = tk.Scrollbar(table_frame, orient="vertical",
                                   command=self.pos_canvas.yview)
        self.pos_inner = tk.Frame(self.pos_canvas, bg=BG_CARD)
        self.pos_inner.bind("<Configure>",
                             lambda e: self.pos_canvas.configure(
                                 scrollregion=self.pos_canvas.bbox("all")))
        self.pos_canvas.create_window((0, 0), window=self.pos_inner, anchor="nw")
        self.pos_canvas.configure(yscrollcommand=pos_scroll.set)
        self.pos_canvas.pack(side="left", fill="both", expand=True)
        pos_scroll.pack(side="right", fill="y")

    # ══════════════════════════════════════════════════════════════
    # PAGE: PERFORMANCE
    # ══════════════════════════════════════════════════════════════

    def _build_page_performance(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["performance"] = page

        tk.Label(page, text="Performance", font=self.font_h2,
                 fg=TEXT_WHITE, bg=BG_PRIMARY).pack(anchor="w", padx=SP8, pady=(SP6, SP3))

        # Stats row
        stats_row = tk.Frame(page, bg=BG_PRIMARY)
        stats_row.pack(fill="x", padx=SP8, pady=(0, SP3))
        stats_row.columnconfigure((0, 1, 2, 3), weight=1)

        self.perf_wr = self._make_stat_card(stats_row, "Win Rate", "--", GREEN, 0)
        self.perf_pf = self._make_stat_card(stats_row, "Profit Factor", "--", ACCENT, 1)
        self.perf_avg = self._make_stat_card(stats_row, "Avg Trade", "--", BLUE, 2)
        self.perf_cnt = self._make_stat_card(stats_row, "Total Trades", "0", CYAN, 3)

        # Chart area
        self.perf_chart_frame = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                          highlightthickness=1)
        self.perf_chart_frame.pack(fill="both", expand=True, padx=SP8, pady=(0, SP5))

    # ══════════════════════════════════════════════════════════════
    # PAGE: RISK
    # ══════════════════════════════════════════════════════════════

    def _build_page_risk(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["risk"] = page

        tk.Label(page, text="Risk Management", font=self.font_h2,
                 fg=TEXT_WHITE, bg=BG_PRIMARY).pack(anchor="w", padx=SP8, pady=(SP6, SP3))

        # Daily Loss Tier
        tier_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                              highlightthickness=1)
        tier_card.pack(fill="x", padx=SP8, pady=(0, SP3))
        tk.Frame(tier_card, bg=YELLOW, height=3).pack(fill="x")
        tier_inner = tk.Frame(tier_card, bg=BG_CARD)
        tier_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(tier_inner, text="Daily Loss Tiers", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self.risk_tier_label = tk.Label(tier_inner, text="Loading...",
                                         font=self.font_body, fg=TEXT_GRAY,
                                         bg=BG_CARD, justify="left", anchor="w")
        self.risk_tier_label.pack(anchor="w", fill="x", pady=(SP2, 0))

        # Correlation Exposure
        corr_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                              highlightthickness=1)
        corr_card.pack(fill="x", padx=SP8, pady=(0, SP3))
        tk.Frame(corr_card, bg=BLUE, height=3).pack(fill="x")
        corr_inner = tk.Frame(corr_card, bg=BG_CARD)
        corr_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(corr_inner, text="Correlation Exposure", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self.risk_corr_label = tk.Label(corr_inner, text="Loading...",
                                         font=self.font_body, fg=TEXT_GRAY,
                                         bg=BG_CARD, justify="left", anchor="w")
        self.risk_corr_label.pack(anchor="w", fill="x", pady=(SP2, 0))

        # Sizing Pipeline
        sizing_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                highlightthickness=1)
        sizing_card.pack(fill="x", padx=SP8, pady=(0, SP5))
        tk.Frame(sizing_card, bg=ACCENT, height=3).pack(fill="x")
        sizing_inner = tk.Frame(sizing_card, bg=BG_CARD)
        sizing_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(sizing_inner, text="Position Sizing Pipeline", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self.risk_sizing_label = tk.Label(
            sizing_inner,
            text="risk_mult -> pattern_cap(1.25x) -> corr_factor -> daily_loss -> HARD CAP 1.5x",
            font=self.font_mono, fg=TEXT_GRAY, bg=BG_CARD, justify="left", anchor="w")
        self.risk_sizing_label.pack(anchor="w", fill="x", pady=(SP2, 0))

    # ══════════════════════════════════════════════════════════════
    # PAGE: SETTINGS
    # ══════════════════════════════════════════════════════════════

    def _build_page_settings(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["settings"] = page

        tk.Label(page, text="Settings", font=self.font_h2,
                 fg=TEXT_WHITE, bg=BG_PRIMARY).pack(anchor="w", padx=SP8, pady=(SP6, SP3))

        # Model Info
        model_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                               highlightthickness=1)
        model_card.pack(fill="x", padx=SP8, pady=(0, SP3))
        tk.Frame(model_card, bg=ACCENT, height=3).pack(fill="x")
        model_inner = tk.Frame(model_card, bg=BG_CARD)
        model_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(model_inner, text="Neural Model", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self.model_info_label = tk.Label(model_inner, text="Loading...",
                                          font=self.font_body, fg=TEXT_GRAY,
                                          bg=BG_CARD, justify="left", anchor="w")
        self.model_info_label.pack(anchor="w", fill="x", pady=(SP2, 0))

        # Agentic System
        agent_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                               highlightthickness=1)
        agent_card.pack(fill="x", padx=SP8, pady=(0, SP3))
        tk.Frame(agent_card, bg=CYAN, height=3).pack(fill="x")
        agent_inner = tk.Frame(agent_card, bg=BG_CARD)
        agent_inner.pack(fill="x", padx=SP5, pady=SP4)
        tk.Label(agent_inner, text="Agentic System", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")
        self.agentic_label = tk.Label(agent_inner, text="Loading...",
                                       font=self.font_body, fg=TEXT_GRAY,
                                       bg=BG_CARD, justify="left", anchor="w")
        self.agentic_label.pack(anchor="w", fill="x", pady=(SP2, 0))

        # Symbol Toggles
        toggle_card = tk.Frame(page, bg=BG_CARD, highlightbackground=CARD_BORDER,
                                highlightthickness=1)
        toggle_card.pack(fill="x", padx=SP8, pady=(0, SP5))
        tk.Frame(toggle_card, bg=GREEN, height=3).pack(fill="x")

        toggle_header = tk.Frame(toggle_card, bg=BG_CARD)
        toggle_header.pack(fill="x", padx=SP5, pady=(SP4, SP2))
        tk.Label(toggle_header, text="Symbol Trading", font=self.font_h3,
                 fg=TEXT_WHITE, bg=BG_CARD).pack(anchor="w")

        toggle_grid = tk.Frame(toggle_card, bg=BG_CARD)
        toggle_grid.pack(fill="x", padx=SP5, pady=(0, SP4))

        self.symbol_vars: Dict[str, tk.BooleanVar] = {}
        self.symbol_checks: Dict[str, tk.Checkbutton] = {}
        for i, sym in enumerate(ALL_SYMBOLS):
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(
                toggle_grid, text=sym, variable=var,
                font=self.font_body, fg=TEXT_WHITE, bg=BG_CARD,
                selectcolor=BG_CARD_ALT, activebackground=BG_CARD,
                activeforeground=ACCENT, indicatoron=True,
                command=lambda s=sym, v=var: self._toggle_symbol(s, v),
            )
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=SP4, pady=SP1)
            self.symbol_vars[sym] = var
            self.symbol_checks[sym] = cb

    # ══════════════════════════════════════════════════════════════
    # PAGE: LOGS
    # ══════════════════════════════════════════════════════════════

    def _build_page_logs(self):
        page = tk.Frame(self.main_area, bg=BG_PRIMARY)
        self.pages["logs"] = page

        # Header with controls
        header = tk.Frame(page, bg=BG_PRIMARY)
        header.pack(fill="x", padx=SP8, pady=(SP6, SP3))
        tk.Label(header, text="System Logs", font=self.font_h2,
                 fg=TEXT_WHITE, bg=BG_PRIMARY).pack(side="left")

        # Controls
        controls = tk.Frame(header, bg=BG_PRIMARY)
        controls.pack(side="right")

        tk.Button(controls, text="Clear", font=self.font_tiny,
                  fg=TEXT_WHITE, bg=BG_CARD, activebackground=BG_CARD_ALT,
                  relief="flat", padx=SP3, command=self._clear_logs).pack(side="right", padx=SP1)

        self.log_search_var = tk.StringVar()
        search_entry = tk.Entry(controls, textvariable=self.log_search_var,
                                 font=self.font_small, fg=TEXT_WHITE, bg=BG_INPUT,
                                 insertbackground=TEXT_WHITE, relief="flat", width=20)
        search_entry.pack(side="right", padx=SP1)

        self.log_level_var = tk.StringVar(value="ALL")
        level_menu = tk.OptionMenu(controls, self.log_level_var,
                                    "ALL", "ERROR", "WARNING", "INFO", "DEBUG")
        level_menu.configure(font=self.font_tiny, fg=TEXT_WHITE, bg=BG_CARD,
                              activebackground=BG_CARD_ALT, highlightthickness=0, relief="flat")
        level_menu.pack(side="right", padx=SP1)

        # Log text widget
        self.log_text = tk.Text(page, font=self.font_mono, bg=BG_CARD,
                                 fg=TEXT_GRAY, insertbackground=TEXT_WHITE,
                                 relief="flat", state="disabled", wrap="none",
                                 highlightbackground=CARD_BORDER, highlightthickness=1)
        self.log_text.pack(fill="both", expand=True, padx=SP8, pady=(0, SP5))

        # Color tags
        self.log_text.tag_configure("ERROR", foreground=RED)
        self.log_text.tag_configure("WARNING", foreground=YELLOW)
        self.log_text.tag_configure("INFO", foreground="#BBBBCC")
        self.log_text.tag_configure("DEBUG", foreground=TEXT_MUTED)

        # Scrollbar
        log_scroll = tk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")

    # ══════════════════════════════════════════════════════════════
    # STARTUP
    # ══════════════════════════════════════════════════════════════

    def _connect_and_start(self):
        def _worker():
            try:
                self._set_status("Connecting to MT5...", YELLOW)
                self.mt5 = MT5Connector()
                if not self.mt5.connect():
                    self._set_status("MT5 OFFLINE", RED)
                    return

                self._set_status("Loading model...", YELLOW)
                self.model_mgr = NeuralModelManager()
                if not self.model_mgr.load_model(MODEL_PATH):
                    self._set_status("Model load failed", RED)
                    return

                self._set_status("Starting engine...", YELLOW)
                self.engine = TradingEngine(
                    mt5_connector=self.mt5,
                    model_manager=self.model_mgr,
                    risk_per_trade=0.08,
                    confidence_threshold=0.65,
                    trading_pairs=ALL_SYMBOLS,
                    max_concurrent_positions=8,
                )

                self.orchestrator = AgenticOrchestrator(
                    model_manager=self.model_mgr,
                    trading_engine=self.engine,
                    model_path=MODEL_PATH,
                    symbols=ALL_SYMBOLS,
                )
                self.engine.orchestrator = self.orchestrator
                self.orchestrator.start()
                self.engine.start()
                self._is_live = True
                self.journal = self.orchestrator.journal

                self._set_status("LIVE", GREEN)
                self.after(0, lambda: self.conn_badge.configure(
                    text="  LIVE  ", bg=GREEN, fg=BG_PRIMARY))
                self._update_model_info()
                self.after(0, self._sync_symbol_switches)

            except Exception as e:
                self._set_status(f"Error: {e}", RED)

        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════
    # PERIODIC UPDATES
    # ══════════════════════════════════════════════════════════════

    def _tick_fast(self):
        if self._stop_event.is_set():
            return
        try:
            self._update_account()
            if self._current_page == "positions":
                self._update_positions_page()
            if self._current_page == "risk":
                self._update_risk_page()
            self._update_dashboard_trades()
        except Exception:
            pass
        self.after(2000, self._tick_fast)

    def _tick_slow(self):
        if self._stop_event.is_set():
            return
        try:
            self._update_performance_page()
            self._update_equity_chart()
            self._update_agentic_status()
        except Exception:
            pass
        self.after(15000, self._tick_slow)

    def _tick_logs(self):
        if self._stop_event.is_set():
            return
        self._drain_logs()
        self.after(500, self._tick_logs)

    # ══════════════════════════════════════════════════════════════
    # DATA UPDATES
    # ══════════════════════════════════════════════════════════════

    def _update_account(self):
        if not self.mt5 or not self._is_live:
            return
        info = self.mt5.get_account_info()
        if not info:
            return
        equity = float(info.get("equity", 0))
        balance = float(info.get("balance", 0))

        self.lbl_account_value.configure(text=f"${equity:,.2f}")
        change_pct = ((equity - balance) / balance * 100) if balance > 0 else 0
        change_color = GREEN if change_pct >= 0 else RED
        self.lbl_account_change.configure(text=f"{change_pct:+.2f}% today", fg=change_color)

        if self.engine:
            daily = float(self.engine.performance_metrics.get("daily_pnl", 0))
            dd = float(self.engine.performance_metrics.get("current_drawdown", 0))

            total_unrealized = sum(
                float(getattr(p, "unrealized_pnl", 0))
                for p in self.engine.positions.values()
                if getattr(p, "status", "OPEN") == "OPEN"
            )
            self.card_unrealized.configure(
                text=f"${total_unrealized:+,.2f}",
                fg=GREEN if total_unrealized >= 0 else RED)
            self.card_daily.configure(
                text=f"${daily:+,.2f}",
                fg=GREEN if daily >= 0 else RED)
            self.card_drawdown.configure(
                text=f"{dd:.2%}",
                fg=RED if dd > 0.05 else YELLOW if dd > 0.02 else GREEN)

            # Update risk bar
            risk_level = min(dd / 0.15, 1.0) if dd > 0 else 0.05
            self._update_risk_bar(risk_level)

    def _update_risk_bar(self, level):
        self._risk_bar_value = level
        try:
            w = self._risk_bar_canvas.winfo_width()
            fill_w = max(4, int(w * level))
            self._risk_bar_canvas.coords(self._risk_bar_fill, 0, 0, fill_w, 8)
            if level < 0.3:
                color, text = GREEN, "Low Risk"
            elif level < 0.6:
                color, text = YELLOW, "Medium Risk"
            else:
                color, text = RED, "High Risk"
            self._risk_bar_canvas.itemconfigure(self._risk_bar_fill, fill=color)
            self._risk_label.configure(text=text, fg=color)
        except Exception:
            pass

    def _update_dashboard_trades(self):
        if not self.engine:
            return
        for w in self.trades_inner.winfo_children():
            w.destroy()

        positions = [p for p in self.engine.positions.values()
                      if getattr(p, "status", "OPEN") == "OPEN"]
        self.trade_count_lbl.configure(text=f"{len(positions)} open")

        for pos in positions[:10]:
            pnl = float(getattr(pos, "unrealized_pnl", 0))
            current = float(getattr(pos, "current_price", pos.entry_price))
            self._make_trade_card(
                self.trades_inner, pos.symbol, pos.action, pnl,
                pos.position_size, pos.entry_price, current)

    def _update_positions_page(self):
        if not self.engine:
            return
        for w in self.pos_inner.winfo_children():
            w.destroy()

        positions = [p for p in self.engine.positions.values()
                      if getattr(p, "status", "OPEN") == "OPEN"]
        self.pos_count_label.configure(text=f"{len(positions)} positions")

        for pos in positions:
            pnl = float(getattr(pos, "unrealized_pnl", 0))
            current = float(getattr(pos, "current_price", pos.entry_price))
            pnl_color = GREEN if pnl >= 0 else RED
            dir_color = GREEN if pos.action == "BUY" else RED

            row = tk.Frame(self.pos_inner, bg=BG_CARD)
            row.pack(fill="x")

            # Zebra stripe
            vals = [
                (pos.symbol, TEXT_WHITE),
                (pos.action, dir_color),
                (f"{pos.entry_price:.5f}", TEXT_GRAY),
                (f"{current:.5f}", TEXT_GRAY),
                (f"{pos.stop_loss:.5f}", TEXT_MUTED),
                (f"{pos.take_profit:.5f}", TEXT_MUTED),
                (f"{pos.position_size:.2f}", TEXT_WHITE),
                (f"${pnl:+.2f}", pnl_color),
            ]
            for txt, color in vals:
                tk.Label(row, text=txt, font=self.font_small, fg=color,
                         bg=BG_CARD, width=12, anchor="w").pack(
                    side="left", padx=SP3, pady=SP1)

            # Bottom separator
            tk.Frame(self.pos_inner, bg=BORDER, height=1).pack(fill="x")

    def _update_risk_page(self):
        if not self.engine:
            return
        factor = self.engine._daily_loss_size_factor
        daily = float(self.engine.performance_metrics.get("daily_pnl", 0))
        start_eq = getattr(self.engine, "_daily_start_equity", 0)
        loss_pct = abs(daily) / start_eq if start_eq > 0 and daily < 0 else 0

        tier_color = GREEN if factor >= 0.9 else YELLOW if factor >= 0.25 else RED
        self.risk_tier_label.configure(
            text=f"Current Size Factor: {factor:.0%}   |   Daily Loss: {loss_pct:.2%}\n\n"
                 f"  5% loss  ->  50% size\n"
                 f"  8% loss  ->  25% size\n"
                 f"  12% loss ->  10% size\n"
                 f"  15% loss ->  FULL STOP",
            fg=tier_color)

        corr_lines = []
        for gname, members in self.engine.correlation_groups.items():
            open_syms = []
            for pos in self.engine.positions.values():
                if getattr(pos, "status", "OPEN") != "OPEN":
                    continue
                ps = re.sub(r"[^A-Z0-9]", "", str(pos.symbol or "").upper())
                if ps in members:
                    sign = +1 if pos.action == "BUY" else -1
                    net = sign * members[ps]
                    d = "LONG" if net > 0 else "SHORT"
                    open_syms.append(f"{ps} {pos.action} ({d})")
            if open_syms:
                corr_lines.append(f"{gname}:\n  " + "\n  ".join(open_syms))
        self.risk_corr_label.configure(
            text="\n".join(corr_lines) if corr_lines else "No correlated positions open")

    def _update_performance_page(self):
        if not self.journal:
            return
        try:
            stats = self.journal.get_rolling_stats(50)
        except Exception:
            return
        wr = stats.get("win_rate", 0)
        pf = stats.get("profit_factor", 0)
        avg = stats.get("avg_pnl_pct", 0)
        cnt = stats.get("trade_count", 0)

        self.perf_wr.configure(text=f"{wr:.1%}", fg=GREEN if wr > 0.50 else RED)
        self.perf_pf.configure(text=f"{pf:.2f}", fg=GREEN if pf > 1.0 else RED)
        self.perf_avg.configure(text=f"{avg:.3%}", fg=GREEN if avg > 0 else RED)
        self.perf_cnt.configure(text=str(cnt))

    def _update_equity_chart(self):
        if not HAS_MATPLOTLIB or not self.journal:
            return
        try:
            trades = self.journal.get_recent_trades(200)
        except Exception:
            return
        if not trades:
            return

        trades_sorted = sorted(trades, key=lambda t: t.exit_time)
        cum = []
        running = 0.0
        for t in trades_sorted:
            running += t.pnl
            cum.append(running)

        # Dashboard chart
        self._render_chart(self.chart_container, cum)
        if self.chart_placeholder:
            try:
                self.chart_placeholder.destroy()
            except Exception:
                pass
            self.chart_placeholder = None

        # Performance page chart
        self._render_chart(self.perf_chart_frame, cum)

    def _render_chart(self, parent, cum_pnl):
        for w in parent.winfo_children():
            w.destroy()

        fig = Figure(figsize=(8, 3.5), dpi=90, facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_CARD)

        x = list(range(len(cum_pnl)))
        ax.plot(x, cum_pnl, color=ACCENT, linewidth=2.5, alpha=0.95)
        ax.fill_between(x, cum_pnl, 0,
                         where=[p >= 0 for p in cum_pnl], color=ACCENT, alpha=0.06)
        ax.fill_between(x, cum_pnl, 0,
                         where=[p < 0 for p in cum_pnl], color=RED, alpha=0.06)
        ax.axhline(y=0, color="#333355", linewidth=0.5, linestyle="--")

        ax.tick_params(colors="#555577", labelsize=8)
        ax.set_ylabel("Cumulative P/L ($)", color="#777799", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#1E1E3A", linewidth=0.3)
        fig.tight_layout(pad=2)

        if cum_pnl:
            last = cum_pnl[-1]
            color = ACCENT if last >= 0 else RED
            txt = f"+${last:,.2f}" if last >= 0 else f"-${abs(last):,.2f}"
            ax.annotate(txt, xy=(len(cum_pnl) - 1, last), fontsize=11, color=color,
                        fontweight="bold", ha="right",
                        xytext=(-10, 12), textcoords="offset points")

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_model_info(self):
        if not self.model_mgr:
            return
        meta = getattr(self.model_mgr, "model_metadata", {}) or {}
        lines = [
            f"Features: {meta.get('feature_dim', '?')}   |   Horizon: {meta.get('horizon', '?')} bars",
            f"Profit Factor: {meta.get('profit_factor', '?')}   |   Expectancy: {meta.get('expectancy', '?')}",
            f"Training: {str(meta.get('training_date', '?'))[:10]}   |   Symbols: {meta.get('num_symbols', '?')}",
            f"Threshold: {meta.get('global_threshold', '?')}",
        ]
        self.after(0, lambda: self.model_info_label.configure(text="\n".join(lines)))

    def _update_agentic_status(self):
        if not self.orchestrator:
            return
        try:
            s = self.orchestrator.get_status()
        except Exception:
            return
        lines = [
            f"Running: {'YES' if s.get('running') else 'NO'}   |   Trades: {s.get('total_journal_trades', 0)}",
            f"Probation: {s.get('probation_active', False)}",
        ]
        self.agentic_label.configure(text="\n".join(lines))

    # ══════════════════════════════════════════════════════════════
    # LOGS
    # ══════════════════════════════════════════════════════════════

    def _drain_logs(self):
        level_map = {"ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "ALL": 0}
        min_lvl = level_map.get(self.log_level_var.get(), 0)
        search = self.log_search_var.get().lower()
        batch = []
        try:
            for _ in range(200):
                lvl, msg = self.log_handler.log_queue.get_nowait()
                if lvl < min_lvl:
                    continue
                if search and search not in msg.lower():
                    continue
                tag = "ERROR" if lvl >= 40 else "WARNING" if lvl >= 30 else "INFO" if lvl >= 20 else "DEBUG"
                batch.append((msg, tag))
        except queue.Empty:
            pass
        if batch:
            self.log_text.configure(state="normal")
            for msg, tag in batch:
                self.log_text.insert("end", msg + "\n", tag)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def _clear_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════
    # SYMBOL TOGGLES
    # ══════════════════════════════════════════════════════════════

    def _toggle_symbol(self, symbol: str, var: tk.BooleanVar):
        if not self.engine:
            return
        if not var.get():
            self.engine.excluded_symbols.add(symbol)
        else:
            self.engine.excluded_symbols.discard(symbol)

    def _sync_symbol_switches(self):
        if not self.engine:
            return
        for sym, var in self.symbol_vars.items():
            if sym in self.engine.excluded_symbols:
                var.set(False)

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    def _set_status(self, text: str, color: str = TEXT_MUTED):
        def _update():
            self.status_label.configure(text=text, fg=color)
            self.status_dot_canvas.delete("all")
            self.status_dot_canvas.create_oval(1, 1, 7, 7, fill=color, outline="")
        self.after(0, _update)

    def _on_close(self):
        self._stop_event.set()
        try:
            if self.orchestrator:
                self.orchestrator.stop()
            if self.engine:
                self.engine.stop()
        except Exception:
            pass
        self.destroy()


def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler("logs/xeo_dashboard.log"),
            logging.StreamHandler(),
        ],
    )
    app = XeoDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()

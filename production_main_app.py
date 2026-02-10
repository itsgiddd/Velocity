#!/usr/bin/env python3
"""
Neural Forex Trading App — Clean Production UI
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.trading_engine import TradingEngine
from app.model_manager import NeuralModelManager
from app.mt5_connector import MT5Connector
from app.config_manager import ConfigManager
from agentic_orchestrator import AgenticOrchestrator


class ProductionNeuralTradingApp:
    """Clean, simple production trading app"""

    # All 9 model symbols
    ALL_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'NZDUSD', 'EURJPY', 'GBPJPY', 'BTCUSD',
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Neural Forex Trader")
        self.root.geometry("720x640")
        self.root.resizable(True, True)

        self.setup_logging()

        self.config_manager = ConfigManager()
        self.mt5_connector = MT5Connector()
        self.model_manager = NeuralModelManager()

        self.is_trading = False
        self.trading_engine = None
        self.orchestrator = None

        self._build_ui()
        self._start_live_update()

        self.logger.info("App initialized")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def setup_logging(self):
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logs_dir / "trading.log")
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.logger.addHandler(ch)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.configure('Big.TButton', padding=6)
        style.configure('Status.TLabel', font=('TkDefaultFont', 9))

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill='both', expand=True)

        # ---- Row 0: Status bar (top) ----
        status_frame = ttk.Frame(main)
        status_frame.pack(fill='x', pady=(0, 8))

        self.lbl_mt5 = ttk.Label(status_frame, text="MT5: --", style='Status.TLabel')
        self.lbl_mt5.pack(side='left', padx=(0, 15))
        self.lbl_model = ttk.Label(status_frame, text="Model: --", style='Status.TLabel')
        self.lbl_model.pack(side='left', padx=(0, 15))
        self.lbl_trading = ttk.Label(status_frame, text="Trading: OFF", style='Status.TLabel')
        self.lbl_trading.pack(side='left', padx=(0, 15))
        self.lbl_balance = ttk.Label(status_frame, text="Balance: --", style='Status.TLabel',
                                     font=('TkDefaultFont', 10, 'bold'))
        self.lbl_balance.pack(side='right')

        ttk.Separator(main, orient='horizontal').pack(fill='x', pady=4)

        # ---- Row 1: Action buttons ----
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill='x', pady=4)

        self.btn_connect = ttk.Button(btn_frame, text="Connect MT5", command=self.connect_mt5,
                                      style='Big.TButton')
        self.btn_connect.pack(side='left', padx=2)

        self.btn_load = ttk.Button(btn_frame, text="Load Model", command=self.load_neural_model,
                                   style='Big.TButton')
        self.btn_load.pack(side='left', padx=2)

        self.btn_start = ttk.Button(btn_frame, text="Start", command=self.start_trading,
                                    style='Big.TButton', state='disabled')
        self.btn_start.pack(side='left', padx=2)

        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self.stop_trading,
                                   style='Big.TButton', state='disabled')
        self.btn_stop.pack(side='left', padx=2)

        self.btn_emergency = ttk.Button(btn_frame, text="Emergency Stop", command=self.emergency_stop,
                                        style='Big.TButton')
        self.btn_emergency.pack(side='right', padx=2)

        ttk.Separator(main, orient='horizontal').pack(fill='x', pady=4)

        # ---- Row 2: Settings (two columns) ----
        settings_frame = ttk.Frame(main)
        settings_frame.pack(fill='x', pady=4)

        # Left column: Mode & risk
        left = ttk.LabelFrame(settings_frame, text="Mode", padding=6)
        left.pack(side='left', fill='both', expand=True, padx=(0, 4))

        self.zp_pure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="ZeroPoint Pure Mode", variable=self.zp_pure_var).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=2)

        ttk.Label(left, text="Fixed Lots:").grid(row=1, column=0, sticky='w', pady=1)
        self.zp_lot_var = tk.StringVar(value="0.40")
        ttk.Entry(left, textvariable=self.zp_lot_var, width=7).grid(row=1, column=1, sticky='w', padx=4)

        ttk.Label(left, text="Risk %:").grid(row=2, column=0, sticky='w', pady=1)
        self.risk_var = tk.StringVar(value="8")
        ttk.Entry(left, textvariable=self.risk_var, width=7).grid(row=2, column=1, sticky='w', padx=4)

        ttk.Label(left, text="Confidence %:").grid(row=3, column=0, sticky='w', pady=1)
        self.confidence_var = tk.StringVar(value="65")
        ttk.Entry(left, textvariable=self.confidence_var, width=7).grid(row=3, column=1, sticky='w', padx=4)

        # Right column: Trade monitor
        right = ttk.LabelFrame(settings_frame, text="Trade Monitor", padding=6)
        right.pack(side='left', fill='both', expand=True, padx=(4, 0))

        ttk.Label(right, text="Max Loss ($):").grid(row=0, column=0, sticky='w', pady=1)
        self.zp_max_loss_var = tk.StringVar(value="80")
        ttk.Entry(right, textvariable=self.zp_max_loss_var, width=7).grid(row=0, column=1, sticky='w', padx=4)

        ttk.Label(right, text="BE Pips:").grid(row=1, column=0, sticky='w', pady=1)
        self.zp_be_pips_var = tk.StringVar(value="15")
        ttk.Entry(right, textvariable=self.zp_be_pips_var, width=7).grid(row=1, column=1, sticky='w', padx=4)

        ttk.Label(right, text="Stall (min):").grid(row=2, column=0, sticky='w', pady=1)
        self.zp_stall_var = tk.StringVar(value="30")
        ttk.Entry(right, textvariable=self.zp_stall_var, width=7).grid(row=2, column=1, sticky='w', padx=4)

        ttk.Label(right, text="Deadline (min):").grid(row=3, column=0, sticky='w', pady=1)
        self.zp_deadline_var = tk.StringVar(value="60")
        ttk.Entry(right, textvariable=self.zp_deadline_var, width=7).grid(row=3, column=1, sticky='w', padx=4)

        # ---- Row 3: Pairs (compact row of checkboxes) ----
        pairs_frame = ttk.LabelFrame(main, text="Pairs", padding=4)
        pairs_frame.pack(fill='x', pady=4)

        self.pair_vars = {}
        for i, pair in enumerate(self.ALL_PAIRS):
            var = tk.BooleanVar(value=(pair != 'USDJPY'))  # USDJPY off by default
            self.pair_vars[pair] = var
            ttk.Checkbutton(pairs_frame, text=pair, variable=var).grid(
                row=0, column=i, padx=4, pady=2)

        # ---- Row 4: Live positions table ----
        pos_frame = ttk.LabelFrame(main, text="Open Positions", padding=4)
        pos_frame.pack(fill='both', expand=True, pady=4)

        cols = ('Symbol', 'Dir', 'Lots', 'Entry', 'Current', 'P/L', 'SL', 'TP')
        self.pos_tree = ttk.Treeview(pos_frame, columns=cols, show='headings', height=5)
        for c in cols:
            self.pos_tree.heading(c, text=c)
            w = 80 if c not in ('Symbol', 'P/L') else 70
            self.pos_tree.column(c, width=w, anchor='center')
        self.pos_tree.pack(fill='both', expand=True)

        # ---- Row 5: Log ----
        log_frame = ttk.LabelFrame(main, text="Log", padding=4)
        log_frame.pack(fill='both', expand=True, pady=(4, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=('Consolas', 8))
        self.log_text.pack(fill='both', expand=True)
        self._log("App ready. Connect MT5 and load model to begin.")

    # ------------------------------------------------------------------
    # Live update loop (runs every 2 seconds)
    # ------------------------------------------------------------------
    def _start_live_update(self):
        self._update_status()
        self.root.after(2000, self._start_live_update)

    def _update_status(self):
        # MT5 status
        mt5_ok = self.mt5_connector.is_connected()
        self.lbl_mt5.config(
            text=f"MT5: {'ON' if mt5_ok else 'OFF'}",
            foreground='green' if mt5_ok else 'gray')

        # Model status
        model_ok = self.model_manager.is_model_loaded()
        self.lbl_model.config(
            text=f"Model: {'OK' if model_ok else '--'}",
            foreground='green' if model_ok else 'gray')

        # Trading status
        self.lbl_trading.config(
            text=f"Trading: {'LIVE' if self.is_trading else 'OFF'}",
            foreground='green' if self.is_trading else 'gray')

        # Balance
        if mt5_ok:
            try:
                info = self.mt5_connector.get_account_info()
                if info:
                    bal = float(info.get('balance', 0))
                    eq = float(info.get('equity', 0))
                    self.lbl_balance.config(text=f"${bal:,.2f} (eq: ${eq:,.2f})")
            except Exception:
                pass

        # Enable start button when ready
        if mt5_ok and model_ok and not self.is_trading:
            self.btn_start.config(state='normal')

        # Update positions table
        self._refresh_positions()

    def _refresh_positions(self):
        """Refresh open positions table from MT5"""
        for row in self.pos_tree.get_children():
            self.pos_tree.delete(row)

        if not self.mt5_connector.is_connected():
            return

        try:
            import MetaTrader5 as mt5_lib
            positions = mt5_lib.positions_get()
            if not positions:
                return
            for p in positions:
                direction = "BUY" if p.type == 0 else "SELL"
                pnl_str = f"${p.profit:+.2f}"
                tag = 'profit' if p.profit >= 0 else 'loss'
                self.pos_tree.insert('', 'end', values=(
                    p.symbol, direction, f"{p.volume:.2f}",
                    f"{p.price_open:.5f}", f"{p.price_current:.5f}",
                    pnl_str, f"{p.sl:.5f}", f"{p.tp:.5f}",
                ), tags=(tag,))
            self.pos_tree.tag_configure('profit', foreground='green')
            self.pos_tree.tag_configure('loss', foreground='red')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert('end', f"[{ts}] {msg}\n")
        self.log_text.see('end')

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def connect_mt5(self):
        def _thread():
            try:
                self.btn_connect.config(state='disabled', text="Connecting...")
                success = self.mt5_connector.connect()
                if success:
                    self.lbl_mt5.config(text="MT5: ON", foreground='green')
                    self._log("MT5 connected")
                    self.logger.info("MT5 connected")

                    info = self.mt5_connector.get_account_info()
                    if info:
                        bal = info.get('balance', '?')
                        self._log(f"Account: {info.get('login', '?')} | Balance: ${bal}")
                else:
                    self._log("MT5 connection failed — is MT5 running?")
            except Exception as e:
                self._log(f"MT5 error: {e}")
            finally:
                self.btn_connect.config(state='normal', text="Connect MT5")
        threading.Thread(target=_thread, daemon=True).start()

    def load_neural_model(self):
        def _thread():
            try:
                self.btn_load.config(state='disabled', text="Loading...")
                model_path = self.config_manager.get_config(
                    'trading', 'neural_network.model_path', 'neural_model.pth')
                success = self.model_manager.load_model(model_path)
                if success:
                    self.lbl_model.config(text="Model: OK", foreground='green')
                    meta = getattr(self.model_manager, 'model_metadata', {}) or {}
                    dim = meta.get('feature_dim', '?')
                    self._log(f"Model loaded (features={dim})")
                    self.logger.info("Model loaded")
                else:
                    self._log("Model load failed")
            except Exception as e:
                self._log(f"Model error: {e}")
            finally:
                self.btn_load.config(state='normal', text="Load Model")
        threading.Thread(target=_thread, daemon=True).start()

    def start_trading(self):
        try:
            if not self.mt5_connector.is_connected():
                self._log("Connect MT5 first")
                return
            if not self.model_manager.is_model_loaded():
                self._log("Load model first")
                return

            risk_per_trade = float(self.risk_var.get()) / 100
            confidence_threshold = float(self.confidence_var.get()) / 100
            selected_pairs = [p for p, v in self.pair_vars.items() if v.get()]

            self.trading_engine = TradingEngine(
                mt5_connector=self.mt5_connector,
                model_manager=self.model_manager,
                risk_per_trade=risk_per_trade,
                confidence_threshold=confidence_threshold,
                trading_pairs=selected_pairs,
            )

            # ZeroPoint Pure Mode
            zp_pure = self.zp_pure_var.get()
            self.trading_engine.zeropoint_pure_mode = zp_pure
            if zp_pure:
                try:
                    self.trading_engine.zeropoint_fixed_lot = float(self.zp_lot_var.get())
                except ValueError:
                    self.trading_engine.zeropoint_fixed_lot = 0.40
                # Skip unchecked symbols
                self.trading_engine.zeropoint_skip_symbols = {
                    p for p, v in self.pair_vars.items() if not v.get()
                }
                # Trade monitor settings
                try:
                    self.trading_engine.zp_max_loss_dollars = float(self.zp_max_loss_var.get())
                except ValueError:
                    self.trading_engine.zp_max_loss_dollars = 80.0
                try:
                    self.trading_engine.zp_breakeven_pips = float(self.zp_be_pips_var.get())
                except ValueError:
                    self.trading_engine.zp_breakeven_pips = 15.0
                try:
                    self.trading_engine.zp_stall_minutes = float(self.zp_stall_var.get())
                except ValueError:
                    self.trading_engine.zp_stall_minutes = 30
                try:
                    self.trading_engine.zp_close_deadline_minutes = float(self.zp_deadline_var.get())
                except ValueError:
                    self.trading_engine.zp_close_deadline_minutes = 60

                self._log(
                    f"ZP Pure Mode | Lot: {self.trading_engine.zeropoint_fixed_lot} | "
                    f"Max loss: ${self.trading_engine.zp_max_loss_dollars} | "
                    f"BE: {self.trading_engine.zp_breakeven_pips}pip | "
                    f"Stall: {self.trading_engine.zp_stall_minutes}m | "
                    f"Deadline: {self.trading_engine.zp_close_deadline_minutes}m"
                )

            # Orchestrator
            self.orchestrator = AgenticOrchestrator(
                model_manager=self.model_manager,
                trading_engine=self.trading_engine,
            )
            self.trading_engine.orchestrator = self.orchestrator
            self.orchestrator.start()

            self.trading_engine.start()
            self.is_trading = True

            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='normal')
            self.lbl_trading.config(text="Trading: LIVE", foreground='green')
            mode = "ZP Pure" if zp_pure else "Neural"
            self._log(f"Trading started ({mode}) — {len(selected_pairs)} pairs")
            self.logger.info("Trading started")

        except Exception as e:
            self._log(f"Start error: {e}")
            self.logger.error(f"Start error: {e}")

    def stop_trading(self):
        try:
            if self.orchestrator:
                self.orchestrator.stop()
            if self.trading_engine:
                self.trading_engine.stop()
            self.is_trading = False
            self.btn_start.config(state='normal')
            self.btn_stop.config(state='disabled')
            self.lbl_trading.config(text="Trading: OFF", foreground='gray')
            self._log("Trading stopped")
            self.logger.info("Trading stopped")
        except Exception as e:
            self._log(f"Stop error: {e}")

    def emergency_stop(self):
        try:
            if self.orchestrator:
                self.orchestrator.stop()
            if self.trading_engine:
                self.trading_engine.stop()
            self.is_trading = False
            self.btn_start.config(state='normal')
            self.btn_stop.config(state='disabled')
            self.lbl_trading.config(text="STOPPED", foreground='red')
            self._log("EMERGENCY STOP")
            self.logger.warning("Emergency stop")
        except Exception as e:
            self._log(f"Emergency stop error: {e}")


def main():
    root = tk.Tk()
    app = ProductionNeuralTradingApp(root)
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (720 // 2)
    y = (root.winfo_screenheight() // 2) - (640 // 2)
    root.geometry(f'720x640+{x}+{y}')
    root.mainloop()


if __name__ == "__main__":
    main()

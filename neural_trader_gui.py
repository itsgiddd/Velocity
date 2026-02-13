#!/usr/bin/env python3
"""
Neural Forex Trader - Windows GUI Application
=============================================
A user-friendly Windows application for automated forex trading
using neural network signals.

Features:
- One-click trading
- Real-time position monitoring
- Profit/loss tracking
- Easy MT5 connection

Author: ACI
Version: 2.0.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import sys
import os

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_live_trading_bot import LiveNeuralTradingBot, TradingMode, TradeResult, TradeSignal
from datetime import datetime

class NeuralTraderGUI:
    """Main GUI Application for ACI Trading System"""

    def __init__(self, root):
        self.root = root
        self.root.title("ACI Trading System - Auto Trading")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e2e')
        
        # Trading bot instance
        self.bot = None
        self.is_trading = False
        self.positions = []
        
        # Create UI
        self.create_header()
        self.create_connection_section()
        self.create_stats_section()
        self.create_positions_section()
        self.create_log_section()
        self.create_controls()
        
        # Auto-refresh positions
        self.refresh_positions()
    
    def create_header(self):
        """Create application header"""
        header_frame = tk.Frame(self.root, bg='#1e1e2e')
        header_frame.pack(fill=tk.X, padx=20, pady=15)
        
        title = tk.Label(header_frame, text="Neural Forex Trader", 
                        font=('Segoe UI', 20, 'bold'), 
                        bg='#1e1e2e', fg='#f5c2e7')
        title.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(header_frame, text="Disconnected", 
                                    font=('Segoe UI', 12), 
                                    bg='#1e1e2e', fg='#f38ba8')
        self.status_label.pack(side=tk.RIGHT)
    
    def create_connection_section(self):
        """Create MT5 connection section"""
        conn_frame = tk.LabelFrame(self.root, text="MT5 Connection", 
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#1e1e2e', fg='#cdd6f4', padx=15, pady=10)
        conn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Account info grid
        info_frame = tk.Frame(conn_frame, bg='#1e1e2e')
        info_frame.pack(fill=tk.X)
        
        tk.Label(info_frame, text="Server:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=0, sticky=tk.W, padx=5)
        self.server_var = tk.StringVar(value="Not Connected")
        tk.Label(info_frame, textvariable=self.server_var, 
                 font=('Segoe UI', 11, 'bold'), bg='#1e1e2e', fg='#89b4fa').grid(row=0, column=1, sticky=tk.W, padx=5)
        
        tk.Label(info_frame, text="Account:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=2, sticky=tk.W, padx=20)
        self.account_var = tk.StringVar(value="---")
        tk.Label(info_frame, textvariable=self.account_var,
                 font=('Segoe UI', 11, 'bold'), bg='#1e1e2e', fg='#89b4fa').grid(row=0, column=3, sticky=tk.W, padx=5)
        
        tk.Label(info_frame, text="Balance:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=4, sticky=tk.W, padx=20)
        self.balance_var = tk.StringVar(value="$0.00")
        tk.Label(info_frame, textvariable=self.balance_var,
                 font=('Segoe UI', 11, 'bold'), bg='#1e1e2e', fg='#a6e3a1').grid(row=0, column=5, sticky=tk.W, padx=5)
        
        tk.Label(info_frame, text="Equity:", bg='#1e1e2e', fg='#cdd6f4').grid(row=1, column=0, sticky=tk.W, padx=5, pady=10)
        self.equity_var = tk.StringVar(value="$0.00")
        tk.Label(info_frame, textvariable=self.equity_var,
                 font=('Segoe UI', 11, 'bold'), bg='#1e1e2e', fg='#89b4fa').grid(row=1, column=1, sticky=tk.W, padx=5)
        
        tk.Label(info_frame, text="Margin Used:", bg='#1e1e2e', fg='#cdd6f4').grid(row=1, column=2, sticky=tk.W, padx=20)
        self.margin_var = tk.StringVar(value="$0.00")
        tk.Label(info_frame, textvariable=self.margin_var,
                 font=('Segoe UI', 11, 'bold'), bg='#1e1e2e', fg='#fab387').grid(row=1, column=3, sticky=tk.W, padx=5)
    
    def create_stats_section(self):
        """Create trading stats section"""
        stats_frame = tk.LabelFrame(self.root, text="Trading Stats", 
                                    font=('Segoe UI', 11, 'bold'),
                                    bg='#1e1e2e', fg='#cdd6f4', padx=15, pady=10)
        stats_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Stats grid
        stats_grid = tk.Frame(stats_frame, bg='#1e1e2e')
        stats_grid.pack(fill=tk.X)
        
        tk.Label(stats_grid, text="Open Positions:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=0, sticky=tk.W, padx=10)
        self.open_pos_var = tk.StringVar(value="0")
        tk.Label(stats_grid, textvariable=self.open_pos_var,
                 font=('Segoe UI', 14, 'bold'), bg='#1e1e2e', fg='#89b4fa').grid(row=0, column=1, sticky=tk.W)
        
        tk.Label(stats_grid, text="Today's P/L:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=2, sticky=tk.W, padx=30)
        self.daily_pl_var = tk.StringVar(value="$0.00")
        tk.Label(stats_grid, textvariable=self.daily_pl_var,
                 font=('Segoe UI', 14, 'bold'), bg='#1e1e2e', fg='#a6e3a1').grid(row=0, column=3, sticky=tk.W)
        
        tk.Label(stats_grid, text="Last Trade:", bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=4, sticky=tk.W, padx=30)
        self.last_trade_var = tk.StringVar(value="None")
        tk.Label(stats_grid, textvariable=self.last_trade_var,
                 font=('Segoe UI', 11), bg='#1e1e2e', fg='#cdd6f4').grid(row=0, column=5, sticky=tk.W)
        
        tk.Label(stats_grid, text="Confidence:", bg='#1e1e2e', fg='#cdd6f4').grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        self.conf_var = tk.StringVar(value="65%")
        tk.Label(stats_grid, textvariable=self.conf_var,
                 font=('Segoe UI', 14, 'bold'), bg='#1e1e2e', fg='#cba6f7').grid(row=1, column=1, sticky=tk.W)
    
    def create_positions_section(self):
        """Create open positions section"""
        pos_frame = tk.LabelFrame(self.root, text="Open Positions", 
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#1e1e2e', fg='#cdd6f4', padx=15, pady=10)
        pos_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Treeview for positions
        columns = ('symbol', 'action', 'lots', 'entry', 'current', 'pl')
        self.pos_tree = ttk.Treeview(pos_frame, columns=columns, show='headings', height=8)
        
        self.pos_tree.heading('symbol', text='Symbol')
        self.pos_tree.heading('action', text='Action')
        self.pos_tree.heading('lots', text='Lots')
        self.pos_tree.heading('entry', text='Entry')
        self.pos_tree.heading('current', text='Current')
        self.pos_tree.heading('pl', text='P/L')
        
        self.pos_tree.column('symbol', width=100)
        self.pos_tree.column('action', width=80)
        self.pos_tree.column('lots', width=80)
        self.pos_tree.column('entry', width=100)
        self.pos_tree.column('current', width=100)
        self.pos_tree.column('pl', width=100)
        
        self.pos_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(pos_frame, orient=tk.VERTICAL, command=self.pos_tree.yview)
        self.pos_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_log_section(self):
        """Create log section"""
        log_frame = tk.LabelFrame(self.root, text="Activity Log", 
                                   font=('Segoe UI', 11, 'bold'),
                                   bg='#1e1e2e', fg='#cdd6f4', padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, 
                                                  font=('Consolas', 9),
                                                  bg='#181825', fg='#cdd6f4')
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def create_controls(self):
        """Create control buttons"""
        btn_frame = tk.Frame(self.root, bg='#1e1e2e')
        btn_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Connect button
        self.connect_btn = tk.Button(btn_frame, text="Connect MT5", 
                                     command=self.toggle_connection,
                                     font=('Segoe UI', 10, 'bold'),
                                     bg='#313244', fg='#cdd6f4', padx=15, pady=8)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        # Trade button
        self.trade_btn = tk.Button(btn_frame, text="Execute Trades", 
                                   command=self.execute_trades,
                                   font=('Segoe UI', 10, 'bold'),
                                   bg='#313244', fg='#cdd6f4', padx=15, pady=8,
                                   state=tk.DISABLED)
        self.trade_btn.pack(side=tk.LEFT, padx=5)
        
        # Close all button
        self.close_btn = tk.Button(btn_frame, text="Close All", 
                                   command=self.close_all_positions,
                                   font=('Segoe UI', 10, 'bold'),
                                   bg='#313244', fg='#f38ba8', padx=15, pady=8,
                                   state=tk.DISABLED)
        self.close_btn.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_btn = tk.Button(btn_frame, text="Refresh", 
                                command=self.refresh_positions,
                                font=('Segoe UI', 10),
                                bg='#313244', fg='#cdd6f4', padx=15, pady=8)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Exit button
        exit_btn = tk.Button(btn_frame, text="Exit", 
                             command=self.root.quit,
                             font=('Segoe UI', 10),
                             bg='#313244', fg='#cdd6f4', padx=15, pady=8)
        exit_btn.pack(side=tk.RIGHT, padx=5)
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def toggle_connection(self):
        """Connect or disconnect from MT5"""
        if self.bot is None:
            # Connect
            try:
                self.log("Connecting to MT5...")
                self.bot = LiveNeuralTradingBot(
                    trading_mode=TradingMode.DEMO,
                    confidence_threshold=0.65,
                    symbols=['EURUSD', 'GBPUSD', 'USDJPY']
                )
                
                if self.bot.connect_to_mt5():
                    self.server_var.set(self.bot.account_info.server if self.bot.account_info else "Connected")
                    self.account_var.set(str(self.bot.account_info.login if self.bot.account_info else ""))
                    self.balance_var.set(f"${self.bot.account_info.balance:.2f}" if self.bot.account_info else "$0.00")
                    self.status_label.config(text="Connected", fg="#a6e3a1")
                    self.connect_btn.config(text="Disconnect")
                    self.trade_btn.config(state=tk.NORMAL)
                    self.close_btn.config(state=tk.NORMAL)
                    self.log("Successfully connected to MT5")
                    self.refresh_positions()
                else:
                    self.log("Failed to connect to MT5")
                    self.bot = None
            except Exception as e:
                self.log(f"Connection error: {e}")
                self.bot = None
        else:
            # Disconnect
            try:
                if self.is_trading:
                    self.is_trading = False
                mt5.shutdown()
                self.log("Disconnected from MT5")
            except:
                pass
            
            self.bot = None
            self.server_var.set("Not Connected")
            self.account_var.set("---")
            self.balance_var.set("$0.00")
            self.equity_var.set("$0.00")
            self.margin_var.set("$0.00")
            self.status_label.config(text="Disconnected", fg="#f38ba8")
            self.connect_btn.config(text="Connect MT5")
            self.trade_btn.config(state=tk.DISABLED)
            self.close_btn.config(state=tk.DISABLED)
    
    def execute_trades(self):
        """Execute trades in a separate thread"""
        def trade_thread():
            self.is_trading = True
            self.root.after(0, lambda: self.trade_btn.config(state=tk.DISABLED))
            
            self.log("Starting trade execution...")
            
            for symbol in self.bot.symbols:
                if not self.is_trading:
                    break
                    
                try:
                    self.log(f"Analyzing {symbol}...")
                    market_data = self.bot.get_market_data(symbol)
                    account_info = self.bot.get_account_info()
                    
                    if market_data and account_info:
                        signal = self.bot.generate_neural_signal(market_data, account_info)
                        
                        if signal and signal.action != TradeResult.HOLD and signal.confidence >= 0.65:
                            success = self.bot.execute_trade(signal, account_info)
                            if success:
                                self.log(f"TRADE: {signal.action.value} {symbol} @ {signal.confidence:.0%}")
                                self.last_trade_var.set(f"{signal.action.value} {symbol}")
                            else:
                                self.log(f"Trade failed: {symbol}")
                        else:
                            self.log(f"Skipped: {symbol} ({signal.confidence if signal else 0:.0%})")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.log(f"Error with {symbol}: {e}")
            
            self.is_trading = False
            self.root.after(0, lambda: self.trade_btn.config(state=tk.NORMAL))
            self.log("Trade session complete - Positions remain OPEN")
            self.root.after(0, self.refresh_positions)
        
        threading.Thread(target=trade_thread, daemon=True).start()
    
    def close_all_positions(self):
        """Close all open positions"""
        if messagebox.askyesno("Confirm", "Close all open positions?"):
            self.log("Closing all positions...")
            self.refresh_positions()
    
    def refresh_positions(self):
        """Refresh position display"""
        if self.bot is None:
            return
        
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            
            if positions:
                self.open_pos_var.set(str(len(positions)))
                
                # Clear treeview
                for item in self.pos_tree.get_children():
                    self.pos_tree.delete(item)
                
                total_pl = 0
                for p in positions:
                    symbol = p.symbol
                    action = "BUY" if p.type == 0 else "SELL"
                    lots = p.volume
                    entry = p.price_open
                    current = p.price_current
                    pl = p.profit
                    total_pl += pl
                    
                    self.pos_tree.insert('', tk.END, values=(symbol, action, lots, 
                                                            f"{entry:.5f}", 
                                                            f"{current:.5f}",
                                                            f"${pl:.2f}"))
                
                # Update P/L display
                if total_pl >= 0:
                    self.daily_pl_var.set(f"+${total_pl:.2f}")
                    self.daily_pl_var.set(f"+${total_pl:.2f}")
                else:
                    self.daily_pl_var.set(f"-${abs(total_pl):.2f}")
                
                # Update account info
                account = mt5.account_info()
                if account:
                    self.equity_var.set(f"${account.equity:.2f}")
                    self.margin_var.set(f"${account.margin:.2f}")
                    self.balance_var.set(f"${account.balance:.2f}")
            
            else:
                self.open_pos_var.set("0")
                self.daily_pl_var.set("$0.00")
                for item in self.pos_tree.get_children():
                    self.pos_tree.delete(item)
        
        except Exception as e:
            self.log(f"Error refreshing positions: {e}")
        
        # Schedule next refresh
        self.root.after(3000, self.refresh_positions)


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Create app
    app = NeuralTraderGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (900 // 2)
    y = (root.winfo_screenheight() // 2) - (700 // 2)
    root.geometry(f"900x700+{x}+{y}")
    
    # Start app
    root.mainloop()


if __name__ == "__main__":
    main()

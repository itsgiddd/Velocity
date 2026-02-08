#!/usr/bin/env python3
"""
Ultimate Scalping Dashboard - Real-time Monitoring & Control
========================================================

Comprehensive dashboard for monitoring the ultimate scalping system:
- Real-time performance metrics
- Active trades monitoring
- Risk management controls
- Profit/loss tracking
- Neural network confidence levels
- System health monitoring
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
import os
from datetime import datetime, timedelta
import numpy as np
import MetaTrader5 as mt5
from typing import Dict, List, Any, Optional
import logging

# Import our trading components
from ultimate_scalping_trading_engine import UltimateScalpingTrader
from app.config_manager import ConfigManager

class UltimateScalpingDashboard:
    """
    Ultimate Scalping Dashboard for real-time monitoring and control
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ultimate Scalping System Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.trader = UltimateScalpingTrader()
        self.running = False
        self.update_interval = 2000  # 2 seconds
        
        # Performance data
        self.performance_data = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'daily_profit': 0.0,
            'win_rate': 0.0,
            'active_positions': 0
        }
        
        # Trading signals
        self.current_signals = {}
        
        # Create GUI
        self._create_widgets()
        self._start_monitoring()
        
        logger = logging.getLogger(__name__)
        logger.info("Ultimate Scalping Dashboard initialized")
    
    def _create_widgets(self):
        """Create dashboard widgets"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="ULTIMATE SCALPING SYSTEM", 
                              font=('Arial', 16, 'bold'), fg='#00ff88', bg='#1a1a1a')
        title_label.pack(pady=(0, 10))
        
        # Control Panel
        self._create_control_panel(main_frame)
        
        # Performance Metrics Panel
        self._create_performance_panel(main_frame)
        
        # Signals Panel
        self._create_signals_panel(main_frame)
        
        # Active Trades Panel
        self._create_trades_panel(main_frame)
        
        # Risk Management Panel
        self._create_risk_panel(main_frame)
    
    def _create_control_panel(self, parent):
        """Create control panel"""
        control_frame = tk.LabelFrame(parent, text="System Controls", 
                                      font=('Arial', 12, 'bold'),
                                      fg='#00ff88', bg='#2a2a2a', 
                                      labelanchor='n')
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Buttons
        button_frame = tk.Frame(control_frame, bg='#2a2a2a')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="START SCALPING", 
                                      command=self._start_trading,
                                      bg='#00aa00', fg='white', 
                                      font=('Arial', 10, 'bold'),
                                      width=15, height=2)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(button_frame, text="STOP TRADING", 
                                    command=self._stop_trading,
                                    bg='#aa0000', fg='white', 
                                    font=('Arial', 10, 'bold'),
                                    width=15, height=2, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        self.emergency_button = tk.Button(button_frame, text="EMERGENCY STOP", 
                                         command=self._emergency_stop,
                                         bg='#ff6600', fg='white', 
                                         font=('Arial', 10, 'bold'),
                                         width=15, height=2)
        self.emergency_button.pack(side='left', padx=5)
        
        # Status indicator
        status_frame = tk.Frame(control_frame, bg='#2a2a2a')
        status_frame.pack(pady=5)
        
        tk.Label(status_frame, text="System Status:", font=('Arial', 10), 
                fg='#ffffff', bg='#2a2a2a').pack(side='left')
        
        self.status_label = tk.Label(status_frame, text="STOPPED", 
                                    font=('Arial', 10, 'bold'),
                                    fg='#ff4444', bg='#2a2a2a')
        self.status_label.pack(side='left', padx=(10, 0))
        
        # Time indicator
        self.time_label = tk.Label(status_frame, text="", 
                                  font=('Arial', 10),
                                  fg='#ffffff', bg='#2a2a2a')
        self.time_label.pack(side='right', padx=10)
    
    def _create_performance_panel(self, parent):
        """Create performance metrics panel"""
        perf_frame = tk.LabelFrame(parent, text="Performance Metrics", 
                                   font=('Arial', 12, 'bold'),
                                   fg='#00ff88', bg='#2a2a2a',
                                   labelanchor='n')
        perf_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Metrics grid
        metrics_grid = tk.Frame(perf_frame, bg='#2a2a2a')
        metrics_grid.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create metric labels
        self.metric_labels = {}
        metrics = [
            ('Total Trades', 'total_trades'),
            ('Winning Trades', 'winning_trades'),
            ('Losing Trades', 'losing_trades'),
            ('Win Rate (%)', 'win_rate'),
            ('Total Profit ($)', 'total_profit'),
            ('Daily Profit ($)', 'daily_profit'),
            ('Active Positions', 'active_positions')
        ]
        
        for i, (label, key) in enumerate(metrics):
            row, col = divmod(i, 2)
            
            tk.Label(metrics_grid, text=f"{label}:", 
                    font=('Arial', 10), fg='#ffffff', bg='#2a2a2a',
                    anchor='e').grid(row=row*2, column=col*2, 
                                   sticky='e', padx=5, pady=2)
            
            self.metric_labels[key] = tk.Label(metrics_grid, text="0", 
                                              font=('Arial', 10, 'bold'), 
                                              fg='#00ff88', bg='#2a2a2a',
                                              anchor='w')
            self.metric_labels[key].grid(row=row*2+1, column=col*2, 
                                        sticky='w', padx=5, pady=2)
    
    def _create_signals_panel(self, parent):
        """Create signals panel"""
        signals_frame = tk.LabelFrame(parent, text="Current Signals", 
                                       font=('Arial', 12, 'bold'),
                                       fg='#00ff88', bg='#2a2a2a',
                                       labelanchor='n')
        signals_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Create treeview for signals
        columns = ('Pair', 'Action', 'Confidence', 'Probability', 'Risk Score')
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, 
                                         show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=80, anchor='center')
        
        # Scrollbar
        signals_scrollbar = ttk.Scrollbar(signals_frame, orient='vertical', 
                                         command=self.signals_tree.yview)
        self.signals_tree.configure(yscrollcommand=signals_scrollbar.set)
        
        self.signals_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        signals_scrollbar.pack(side='right', fill='y', pady=10)
    
    def _create_trades_panel(self, parent):
        """Create active trades panel"""
        trades_frame = tk.LabelFrame(parent, text="Active Trades", 
                                     font=('Arial', 12, 'bold'),
                                     fg='#00ff88', bg='#2a2a2a',
                                     labelanchor='n')
        trades_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Create treeview for trades
        columns = ('Symbol', 'Action', 'Entry Price', 'Current Price', 'P&L', 'Time Held', 'Confidence')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, 
                                        show='headings', height=8)
        
        # Configure columns
        for col in columns:
            self.trades_tree.heading(col, text=col)
            if col in ['Symbol', 'Action']:
                self.trades_tree.column(col, width=80, anchor='center')
            else:
                self.trades_tree.column(col, width=100, anchor='center')
        
        # Scrollbar
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient='vertical', 
                                        command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_scrollbar.set)
        
        self.trades_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        trades_scrollbar.pack(side='right', fill='y', pady=10)
    
    def _create_risk_panel(self, parent):
        """Create risk management panel"""
        risk_frame = tk.LabelFrame(parent, text="Risk Management", 
                                   font=('Arial', 12, 'bold'),
                                   fg='#00ff88', bg='#2a2a2a',
                                   labelanchor='n')
        risk_frame.pack(fill='x', pady=(10, 0))
        
        # Risk controls
        controls_frame = tk.Frame(risk_frame, bg='#2a2a2a')
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Confidence threshold
        tk.Label(controls_frame, text="Min Confidence:", 
                font=('Arial', 10), fg='#ffffff', bg='#2a2a2a').pack(side='left')
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(controls_frame, from_=0.1, to=0.9, 
                                   resolution=0.05, orient='horizontal',
                                   variable=self.confidence_var,
                                   bg='#2a2a2a', fg='#ffffff',
                                   highlightthickness=0)
        confidence_scale.pack(side='left', fill='x', expand=True, padx=10)
        
        # Position size
        tk.Label(controls_frame, text="Position Size:", 
                font=('Arial', 10), fg='#ffffff', bg='#2a2a2a').pack(side='left', padx=(20, 5))
        
        self.position_size_var = tk.DoubleVar(value=2.0)
        position_scale = tk.Scale(controls_frame, from_=0.1, to=5.0, 
                                 resolution=0.1, orient='horizontal',
                                 variable=self.position_size_var,
                                 bg='#2a2a2a', fg='#ffffff',
                                 highlightthickness=0)
        position_scale.pack(side='left', fill='x', expand=True, padx=10)
        
        # Update buttons
        update_button = tk.Button(controls_frame, text="Update Settings", 
                                 command=self._update_risk_settings,
                                 bg='#4444aa', fg='white',
                                 font=('Arial', 10, 'bold'))
        update_button.pack(side='right', padx=10)
    
    def _start_trading(self):
        """Start the scalping system"""
        try:
            self.running = True
            self.status_label.config(text="RUNNING", fg='#00ff88')
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
            # Start trading in separate thread
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            messagebox.showinfo("Trading Started", "Ultimate scalping system is now running!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start trading: {str(e)}")
    
    def _stop_trading(self):
        """Stop the scalping system"""
        try:
            self.running = False
            self.status_label.config(text="STOPPED", fg='#ff4444')
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            messagebox.showinfo("Trading Stopped", "Ultimate scalping system has been stopped.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop trading: {str(e)}")
    
    def _emergency_stop(self):
        """Emergency stop all trading"""
        try:
            # Stop all trading immediately
            self.running = False
            
            # Close all active positions (implementation would depend on MT5)
            # For now, just update status
            
            self.status_label.config(text="EMERGENCY STOP", fg='#ff6600')
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            messagebox.showwarning("Emergency Stop", "All trading has been stopped immediately!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Emergency stop failed: {str(e)}")
    
    def _update_risk_settings(self):
        """Update risk management settings"""
        try:
            # Update trader settings
            self.trader.scalping_config['min_confidence'] = self.confidence_var.get()
            self.trader.scalping_config['position_size_multiplier'] = self.position_size_var.get()
            
            messagebox.showinfo("Settings Updated", "Risk management settings have been updated.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update settings: {str(e)}")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Generate signals
                for symbol in self.trader.trading_pairs:
                    signal = self.trader.generate_scalping_signal(symbol)
                    self.current_signals[symbol] = signal
                    
                    # Execute trades
                    if signal['confidence'] > self.confidence_var.get():
                        self.trader.execute_scalping_trade(symbol, signal)
                
                # Update performance data
                self._update_performance_data()
                
                # Sleep for 30 seconds between cycles
                time.sleep(30)
                
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)  # Longer sleep on error
    
    def _update_performance_data(self):
        """Update performance data"""
        try:
            # Get data from trader
            self.performance_data = {
                'total_trades': self.trader.performance_metrics['total_trades'],
                'winning_trades': self.trader.performance_metrics['winning_trades'],
                'losing_trades': self.trader.performance_metrics['losing_trades'],
                'total_profit': self.trader.performance_metrics['total_profit'],
                'daily_profit': self.trader.performance_metrics['daily_profit'],
                'active_positions': len(self.trader.active_positions)
            }
            
            # Calculate win rate
            if self.performance_data['total_trades'] > 0:
                self.performance_data['win_rate'] = (
                    self.performance_data['winning_trades'] / 
                    self.performance_data['total_trades'] * 100
                )
            else:
                self.performance_data['win_rate'] = 0.0
            
        except Exception as e:
            print(f"Error updating performance data: {e}")
    
    def _update_ui(self):
        """Update UI elements"""
        try:
            # Update time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.config(text=current_time)
            
            # Update performance metrics
            for key, label in self.metric_labels.items():
                value = self.performance_data.get(key, 0)
                if key in ['total_profit', 'daily_profit']:
                    label.config(text=f"${value:.2f}")
                elif key == 'win_rate':
                    label.config(text=f"{value:.1f}%")
                else:
                    label.config(text=str(value))
            
            # Update signals tree
            self._update_signals_display()
            
            # Update trades tree
            self._update_trades_display()
            
        except Exception as e:
            print(f"Error updating UI: {e}")
        
        # Schedule next update
        self.root.after(self.update_interval, self._update_ui)
    
    def _update_signals_display(self):
        """Update signals display"""
        try:
            # Clear existing items
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Add current signals
            for symbol, signal in self.current_signals.items():
                # Color code based on confidence
                if signal['confidence'] > 0.7:
                    tags = ('high_confidence',)
                elif signal['confidence'] > 0.5:
                    tags = ('medium_confidence',)
                else:
                    tags = ('low_confidence',)
                
                self.signals_tree.insert('', 'end', values=(
                    symbol,
                    signal.get('action', 'HOLD'),
                    f"{signal.get('confidence', 0):.3f}",
                    f"{signal.get('probability', 0):.3f}",
                    f"{signal.get('risk_score', 0):.3f}"
                ), tags=tags)
            
            # Configure tags for color coding
            self.signals_tree.tag_configure('high_confidence', background='#004400')
            self.signals_tree.tag_configure('medium_confidence', background='#444400')
            self.signals_tree.tag_configure('low_confidence', background='#440000')
            
        except Exception as e:
            print(f"Error updating signals display: {e}")
    
    def _update_trades_display(self):
        """Update trades display"""
        try:
            # Clear existing items
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)
            
            # Add active trades
            for position_id, position in self.trader.active_positions.items():
                # Calculate current P&L (simplified)
                current_price = self.trader._get_current_price(position['symbol'])
                if current_price:
                    entry_price = position['entry_price']
                    if position['action'] == 'BUY':
                        pnl = (current_price - entry_price) * position['position_size'] * 100000
                    else:
                        pnl = (entry_price - current_price) * position['position_size'] * 100000
                    
                    pnl_text = f"${pnl:.2f}"
                    
                    # Color code P&L
                    if pnl > 0:
                        tags = ('profit',)
                    else:
                        tags = ('loss',)
                else:
                    pnl_text = "$0.00"
                    tags = ()
                
                # Calculate time held
                time_held = datetime.now() - position['entry_time']
                time_text = f"{time_held.seconds // 60}m {time_held.seconds % 60}s"
                
                self.trades_tree.insert('', 'end', values=(
                    position['symbol'],
                    position['action'],
                    f"{position['entry_price']:.5f}",
                    f"{current_price:.5f}" if current_price else "N/A",
                    pnl_text,
                    time_text,
                    f"{position['confidence']:.3f}"
                ), tags=tags)
            
            # Configure P&L color coding
            self.trades_tree.tag_configure('profit', background='#004400', foreground='white')
            self.trades_tree.tag_configure('loss', background='#440000', foreground='white')
            
        except Exception as e:
            print(f"Error updating trades display: {e}")
    
    def _start_monitoring(self):
        """Start the monitoring loop"""
        self._update_ui()
    
    def run(self):
        """Run the dashboard"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Error running dashboard: {e}")
    
    def _on_closing(self):
        """Handle window closing"""
        try:
            self.running = False
            self.root.destroy()
        except Exception as e:
            print(f"Error on closing: {e}")

def main():
    """Main function"""
    try:
        # Create and run dashboard
        dashboard = UltimateScalpingDashboard()
        dashboard.run()
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        messagebox.showerror("Error", f"Failed to start dashboard: {str(e)}")

if __name__ == "__main__":
    main()

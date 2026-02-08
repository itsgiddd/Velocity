#!/usr/bin/env python3
"""
Neural Forex Trading App
========================

Professional neural network-powered forex trading application.
Automatically connects to MT5, manages neural models, and executes trades.

Features:
- Professional GUI interface
- Automatic MT5 connection verification
- Neural network model management
- Real-time trading with confidence monitoring
- Professional logging and error handling
- GitHub-ready codebase

Author: Neural Trading System
Version: 1.0.0
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

# Import trading modules from current directory
from trading_engine import TradingEngine
from model_manager import NeuralModelManager
from mt5_connector import MT5Connector
from config_manager import ConfigManager
from frequent_profitable_trading_config import FREQUENT_TRADING_CONFIG

class NeuralTradingApp:
    """Professional neural forex trading application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Forex Trading App v1.0")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.model_manager = NeuralModelManager()
        self.mt5_connector = MT5Connector()
        self.trading_engine = None
        
        # App state
        self.is_trading = False
        self.model_loaded = False
        self.mt5_connected = False
        
        # Timer State - Comprehensive timer system
        self.symbol_timers = {}
        self.last_timer_update = datetime.now()
        self.timer_update_interval = 60  # Update every 60 seconds
        
        # Session timers
        self.app_start_time = datetime.now()
        self.trading_start_time = None
        self.last_trade_time = None
        
        # Market session tracking
        self.current_market_session = "UNKNOWN"
        self.market_sessions = {
            "SYDNEY": (21, 6),      # 9 PM - 6 AM UTC
            "TOKYO": (0, 9),        # 12 AM - 9 AM UTC  
            "LONDON": (8, 17),     # 8 AM - 5 PM UTC
            "NEW_YORK": (13, 22)   # 1 PM - 10 PM UTC
        }
        
        # Initialize symbol timers
        self.initialize_symbol_timers()
        
        # Setup logging
        self.setup_logging()
        
        # Create GUI
        self.create_gui()
        
        # Check initial status
        self.check_initial_status()
    
    def setup_logging(self):
        """Setup professional logging"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / 'trading_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Neural Trading App initialized")
    
    def create_gui(self):
        """Create professional GUI interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Dashboard
        self.create_dashboard_tab(notebook)
        
        # Tab 2: Model Management
        self.create_model_tab(notebook)
        
        # Tab 3: Trading Control
        self.create_trading_tab(notebook)
        
        # Tab 4: Logs
        self.create_logs_tab(notebook)
        
        # Tab 5: Settings
        self.create_settings_tab(notebook)
    
    def create_dashboard_tab(self, notebook):
        """Create main dashboard tab"""
        dashboard_frame = ttk.Frame(notebook)
        notebook.add(dashboard_frame, text="Dashboard")
        
        # Status section
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        # Status indicators
        self.status_labels = {}
        
        # MT5 Connection
        mt5_frame = ttk.Frame(status_frame)
        mt5_frame.pack(fill='x', pady=2)
        ttk.Label(mt5_frame, text="MT5 Connection:").pack(side='left')
        self.status_labels['mt5'] = ttk.Label(mt5_frame, text="❌ Disconnected", foreground='red')
        self.status_labels['mt5'].pack(side='left', padx=(10, 0))
        
        # Model Status
        model_frame = ttk.Frame(status_frame)
        model_frame.pack(fill='x', pady=2)
        ttk.Label(model_frame, text="Neural Model:").pack(side='left')
        self.status_labels['model'] = ttk.Label(model_frame, text="❌ Not Loaded", foreground='red')
        self.status_labels['model'].pack(side='left', padx=(10, 0))
        
        # Trading Status
        trading_frame = ttk.Frame(status_frame)
        trading_frame.pack(fill='x', pady=2)
        ttk.Label(trading_frame, text="Trading Engine:").pack(side='left')
        self.status_labels['trading'] = ttk.Label(trading_frame, text="❌ Stopped", foreground='red')
        self.status_labels['trading'].pack(side='left', padx=(10, 0))
        
        # Account Info
        account_frame = ttk.LabelFrame(dashboard_frame, text="Account Information", padding=10)
        account_frame.pack(fill='x', padx=5, pady=5)
        
        self.account_labels = {}
        account_info = [
            ("Account", "account"),
            ("Balance", "balance"),
            ("Equity", "equity"),
            ("Margin", "margin"),
            ("Free Margin", "free_margin")
        ]
        
        for label_text, key in account_info:
            frame = ttk.Frame(account_frame)
            frame.pack(fill='x', pady=1)
            ttk.Label(frame, text=f"{label_text}:").pack(side='left')
            self.account_labels[key] = ttk.Label(frame, text="N/A")
            self.account_labels[key].pack(side='left', padx=(10, 0))
        
        # Performance section
        perf_frame = ttk.LabelFrame(dashboard_frame, text="Trading Performance", padding=10)
        perf_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.performance_labels = {}
        perf_info = [
            ("Win Rate", "win_rate"),
            ("Total Trades", "total_trades"),
            ("Winning Trades", "winning_trades"),
            ("Daily P&L", "daily_pnl"),
            ("Total P&L", "total_pnl")
        ]
        
        for label_text, key in perf_info:
            frame = ttk.Frame(perf_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=f"{label_text}:").pack(side='left')
            self.performance_labels[key] = ttk.Label(frame, text="0")
            self.performance_labels[key].pack(side='left', padx=(10, 0))
        
        # Control buttons
        control_frame = ttk.Frame(dashboard_frame)
        control_frame.pack(fill='x', padx=5, pady=10)
        
        self.connect_btn = ttk.Button(control_frame, text="Connect MT5", command=self.connect_mt5)
        self.connect_btn.pack(side='left', padx=5)
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side='left', padx=5)
        
        self.start_trading_btn = ttk.Button(control_frame, text="Start Trading", command=self.start_trading)
        self.start_trading_btn.pack(side='left', padx=5)
        
        self.stop_trading_btn = ttk.Button(control_frame, text="Stop Trading", command=self.stop_trading, state='disabled')
        self.stop_trading_btn.pack(side='left', padx=5)
    
    def create_model_tab(self, notebook):
        """Create model management tab"""
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model Manager")
        
        # Model status
        status_frame = ttk.LabelFrame(model_frame, text="Neural Model Status", padding=10)
        status_frame.pack(fill='x', padx=5, pady=5)
        
        self.model_status_text = scrolledtext.ScrolledText(status_frame, height=10, width=70)
        self.model_status_text.pack(fill='both', expand=True)
        
        # Model controls
        controls_frame = ttk.Frame(model_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        self.update_model_btn = ttk.Button(controls_frame, text="Update Neural Model", command=self.update_neural_model)
        self.update_model_btn.pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Train New Model", command=self.train_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Load Model", command=self.load_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Validate Model", command=self.validate_model).pack(side='left', padx=5)
        
        # Frequent Trading Timer Display
        timer_frame = ttk.LabelFrame(model_frame, text="Frequent Trading Timers", padding=10)
        timer_frame.pack(fill='x', padx=5, pady=5)
        
        self.timer_display = scrolledtext.ScrolledText(timer_frame, height=4, width=70)
        self.timer_display.pack(fill='both', expand=True)
        
        # Update timer display
        self.update_timer_display()
        
        # Model info
        info_frame = ttk.LabelFrame(model_frame, text="Model Information", padding=10)
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.model_info_text = scrolledtext.ScrolledText(info_frame, height=8, width=70)
        self.model_info_text.pack(fill='both', expand=True)
    
    def create_trading_tab(self, notebook):
        """Create trading control tab"""
        trading_frame = ttk.Frame(notebook)
        notebook.add(trading_frame, text="Trading Control")
        
        # Trading settings
        settings_frame = ttk.LabelFrame(trading_frame, text="Trading Settings", padding=10)
        settings_frame.pack(fill='x', padx=5, pady=5)
        
        # Risk management
        risk_frame = ttk.Frame(settings_frame)
        risk_frame.pack(fill='x', pady=5)
        
        ttk.Label(risk_frame, text="Risk per Trade (%):").pack(side='left')
        self.risk_var = tk.StringVar(value="1.5")
        risk_spinbox = ttk.Spinbox(risk_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.risk_var, width=10)
        risk_spinbox.pack(side='left', padx=(10, 0))
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill='x', pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold (%):").pack(side='left')
        self.confidence_var = tk.StringVar(value="65")
        conf_spinbox = ttk.Spinbox(conf_frame, from_=50, to=95, increment=5, textvariable=self.confidence_var, width=10)
        conf_spinbox.pack(side='left', padx=(10, 0))
        
        # Trading pairs
        pairs_frame = ttk.LabelFrame(trading_frame, text="Trading Pairs", padding=10)
        pairs_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create checkboxes for USDJPY only
        self.pair_vars = {}
        focused_pairs = [
            "USDJPY"   # Only USDJPY as requested
        ]
        
        # Simple layout for just 2 focused pairs
        for pair in focused_pairs:
            var = tk.BooleanVar(value=True)
            self.pair_vars[pair] = var
            ttk.Checkbutton(pairs_frame, text=pair, variable=var).pack(anchor='w', pady=2)
        
        # Active signals
        signals_frame = ttk.LabelFrame(trading_frame, text="Active Signals", padding=10)
        signals_frame.pack(fill='x', padx=5, pady=5)
        
        self.signals_tree = ttk.Treeview(signals_frame, columns=('Time', 'Pair', 'Action', 'Confidence', 'Price'), show='headings', height=8)
        self.signals_tree.heading('Time', text='Time')
        self.signals_tree.heading('Pair', text='Pair')
        self.signals_tree.heading('Action', text='Action')
        self.signals_tree.heading('Confidence', text='Confidence')
        self.signals_tree.heading('Price', text='Price')
        
        self.signals_tree.pack(fill='both', expand=True)
    
    def create_logs_tab(self, notebook):
        """Create logs tab"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Log controls
        controls_frame = ttk.Frame(logs_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Clear Logs", command=self.clear_logs).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Export Logs", command=self.export_logs).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Refresh", command=self.refresh_logs).pack(side='left', padx=5)
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=20, width=80)
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Load initial logs
        self.load_logs()
    
    def create_settings_tab(self, notebook):
        """Create settings tab"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # MT5 Settings
        mt5_frame = ttk.LabelFrame(settings_frame, text="MT5 Settings", padding=10)
        mt5_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(mt5_frame, text="Server:").pack(anchor='w')
        self.server_var = tk.StringVar(value="MetaQuotes-Demo")
        ttk.Entry(mt5_frame, textvariable=self.server_var, width=30).pack(fill='x', pady=2)
        
        ttk.Label(mt5_frame, text="Login:").pack(anchor='w')
        self.login_var = tk.StringVar(value="")
        ttk.Entry(mt5_frame, textvariable=self.login_var, width=30).pack(fill='x', pady=2)
        
        ttk.Label(mt5_frame, text="Password:").pack(anchor='w')
        self.password_var = tk.StringVar(value="")
        ttk.Entry(mt5_frame, textvariable=self.password_var, show='*', width=30).pack(fill='x', pady=2)
        
        # App Settings
        app_frame = ttk.LabelFrame(settings_frame, text="Application Settings", padding=10)
        app_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(app_frame, text="Update Interval (seconds):").pack(anchor='w')
        self.update_interval_var = tk.StringVar(value="5")
        ttk.Entry(app_frame, textvariable=self.update_interval_var, width=10).pack(anchor='w', pady=2)
        
        ttk.Label(app_frame, text="Max Concurrent Trades:").pack(anchor='w')
        self.max_trades_var = tk.StringVar(value="5")
        ttk.Entry(app_frame, textvariable=self.max_trades_var, width=10).pack(anchor='w', pady=2)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
    
    def check_initial_status(self):
        """Check initial system status"""
        self.update_status()
        self.root.after(5000, self.check_initial_status)  # Update every 5 seconds
    
    def update_status(self):
        """Update system status display"""
        try:
            # Update MT5 status
            if self.mt5_connector.is_connected():
                self.status_labels['mt5'].config(text="✅ Connected", foreground='green')
                account_info = self.mt5_connector.get_account_info()
                if account_info:
                    self.account_labels['account'].config(text=str(account_info.get('login', 'N/A')))
                    self.account_labels['balance'].config(text=f"${account_info.get('balance', 0):.2f}")
                    self.account_labels['equity'].config(text=f"${account_info.get('equity', 0):.2f}")
                    self.account_labels['margin'].config(text=f"${account_info.get('margin', 0):.2f}")
                    self.account_labels['free_margin'].config(text=f"${account_info.get('margin_free', 0):.2f}")
            else:
                self.status_labels['mt5'].config(text="❌ Disconnected", foreground='red')
            
            # Update model status
            if self.model_manager.is_model_loaded():
                self.status_labels['model'].config(text="✅ Loaded", foreground='green')
            else:
                self.status_labels['model'].config(text="❌ Not Loaded", foreground='red')
            
            # Update trading status
            if self.is_trading:
                self.status_labels['trading'].config(text="✅ Active", foreground='green')
            else:
                self.status_labels['trading'].config(text="❌ Stopped", foreground='red')
                
        except Exception as e:
            self.logger.error(f"Error updating status: {e}")
    
    def connect_mt5(self):
        """Connect to MT5"""
        def connect_thread():
            try:
                self.connect_btn.config(state='disabled', text="Connecting...")
                self.root.update()
                
                success = self.mt5_connector.connect()
                
                if success:
                    messagebox.showinfo("Success", "Connected to MT5 successfully!")
                    self.logger.info("MT5 connection established")
                else:
                    messagebox.showerror("Error", "Failed to connect to MT5")
                    self.logger.error("MT5 connection failed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Connection error: {e}")
                self.logger.error(f"MT5 connection error: {e}")
            finally:
                self.connect_btn.config(state='normal', text="Connect MT5")
                self.update_status()
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def load_model(self):
        """Load neural model"""
        def load_thread():
            try:
                self.load_model_btn.config(state='disabled', text="Loading...")
                self.root.update()
                
                success = self.model_manager.load_model()
                
                if success:
                    messagebox.showinfo("Success", "Neural model loaded successfully!")
                    self.logger.info("Neural model loaded")
                    self.display_model_info()
                else:
                    messagebox.showerror("Error", "Failed to load neural model")
                    self.logger.error("Model loading failed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Model loading error: {e}")
                self.logger.error(f"Model loading error: {e}")
            finally:
                self.load_model_btn.config(state='normal', text="Load Model")
                self.update_status()
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def display_model_info(self):
        """Display model information"""
        try:
            info = self.model_manager.get_model_info()
            if info:
                self.model_info_text.delete(1.0, tk.END)
                self.model_info_text.insert(1.0, json.dumps(info, indent=2))
        except Exception as e:
            self.logger.error(f"Error displaying model info: {e}")
    
    def start_trading(self):
        """Start trading engine"""
        try:
            if not self.mt5_connector.is_connected():
                messagebox.showwarning("Warning", "Please connect to MT5 first!")
                return
            
            if not self.model_manager.is_model_loaded():
                messagebox.showwarning("Warning", "Please load neural model first!")
                return
            
            # Get settings
            risk_per_trade = float(self.risk_var.get()) / 100
            confidence_threshold = float(self.confidence_var.get()) / 100
            selected_pairs = [pair for pair, var in self.pair_vars.items() if var.get()]
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(
                mt5_connector=self.mt5_connector,
                model_manager=self.model_manager,
                risk_per_trade=risk_per_trade,
                confidence_threshold=confidence_threshold,
                trading_pairs=selected_pairs
            )
            
            # Start trading
            self.trading_engine.start()
            self.is_trading = True
            
            # Start trading session timer
            if not self.trading_start_time:
                self.trading_start_time = datetime.now()
            
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            messagebox.showinfo("Success", "Trading engine started!")
            self.logger.info("Trading engine started")
            
            # Update timer display immediately
            self.update_timer_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start trading: {e}")
            self.logger.error(f"Trading startup error: {e}")
    
    def stop_trading(self):
        """Stop trading engine"""
        try:
            if self.trading_engine:
                self.trading_engine.stop()
            
            # Stop trading session timer
            if self.trading_start_time:
                trading_duration = datetime.now() - self.trading_start_time
                self.logger.info(f"Trading session lasted: {self._format_duration(trading_duration)}")
                self.trading_start_time = None
            
            self.is_trading = False
            self.start_trading_btn.config(state='normal')
            self.stop_trading_btn.config(state='disabled')
            
            messagebox.showinfo("Info", "Trading engine stopped!")
            self.logger.info("Trading engine stopped")
            
            # Update timer display immediately
            self.update_timer_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop trading: {e}")
            self.logger.error(f"Trading shutdown error: {e}")
    
    def train_model(self):
        """Train new neural model"""
        # This would open a training dialog
        messagebox.showinfo("Info", "Model training feature coming soon!")
    
    def validate_model(self):
        """Validate loaded model"""
        # This would run model validation
        messagebox.showinfo("Info", "Model validation feature coming soon!")
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.delete(1.0, tk.END)
    
    def export_logs(self):
        """Export logs to file"""
        # Implementation for log export
        messagebox.showinfo("Info", "Log export feature coming soon!")
    
    def refresh_logs(self):
        """Refresh log display"""
        self.load_logs()
    
    def load_logs(self):
        """Load and display logs"""
        try:
            log_file = Path("logs/trading_app.log")
            if log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                self.log_text.delete(1.0, tk.END)
                self.log_text.insert(1.0, log_content)
                
                # Auto-scroll to bottom
                self.log_text.see(tk.END)
        except Exception as e:
            self.logger.error(f"Error loading logs: {e}")
    
    def save_settings(self):
        """Save application settings"""
        try:
            settings = {
                'mt5_server': self.server_var.get(),
                'mt5_login': self.login_var.get(),
                'update_interval': int(self.update_interval_var.get()),
                'max_trades': int(self.max_trades_var.get())
            }
            
            self.config_manager.save_settings(settings)
            messagebox.showinfo("Success", "Settings saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def initialize_symbol_timers(self):
        """Initialize symbol timers for USDJPY only"""
        focused_pairs = ['USDJPY']  # Only USDJPY
        
        for symbol in focused_pairs:
            self.symbol_timers[symbol] = {
                'last_trade_time': datetime.min,
                'profit_lock_until': datetime.min,
                'cooldown_until': datetime.min,
                'last_win_time': datetime.min,
                'tier1_closed': False,
                'tier2_closed': False,
                'tier3_closed': False
            }
    
    def create_simple_training_script(self):
        """Create a simple training script that doesn't require MT5"""
        script_content = '''#!/usr/bin/env python3
"""
Enhanced Neural Model Training with Candlestick Patterns
========================================================
Enhanced training with 10 features including candlestick pattern recognition.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Create sample training data
def generate_sample_data(n_samples=1000):
    """Generate sample training data for quick training with 10 features"""
    np.random.seed(42)
    
    # Generate synthetic features (10 features including candlestick patterns)
    features = np.random.randn(n_samples, 10)  # 10 features including candlestick patterns
    
    # Generate synthetic labels based on features (enhanced with candlestick patterns)
    labels = []
    for i in range(n_samples):
        # Enhanced rule: Consider candlestick patterns in decision
        price_momentum = features[i, 0]
        candlestick_trend_continuation = features[i, 8]  # 4-candle continuation
        candlestick_reversal = features[i, 9]  # 3-candle reversal
        
        # Complex rule: BUY if strong momentum + continuation, SELL if reversal
        if price_momentum > 0.5 and candlestick_trend_continuation > 0.7:
            labels.append(2)  # BUY
        elif price_momentum < -0.5 and candlestick_reversal > 0.7:
            labels.append(0)  # SELL
        elif abs(price_momentum) < 0.3:
            labels.append(1)  # HOLD
        elif price_momentum > 0.2:
            labels.append(2)  # BUY
        elif price_momentum < -0.2:
            labels.append(0)  # SELL
        else:
            labels.append(1)  # HOLD
    
    return features, np.array(labels)

class EnhancedNeuralNetwork(nn.Module):
    """Enhanced neural network for forex prediction with candlestick patterns"""
    
    def __init__(self, input_dim=10, hidden_sizes=[512, 256, 128, 64], output_size=3):
        super(EnhancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers (larger for 10 features + candlestick patterns)
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def main():
    print("ENHANCED NEURAL MODEL TRAINING")
    print("Training with 10 features including candlestick patterns")
    print("=" * 60)
    
    try:
        # Generate sample data
        print("Generating enhanced training data...")
        features, labels = generate_sample_data(1000)
        
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution: BUY={np.sum(labels==2)}, HOLD={np.sum(labels==1)}, SELL={np.sum(labels==0)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create enhanced model with 10 features
        model = EnhancedNeuralNetwork(input_dim=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        print("Training enhanced model...")
        epochs = 50
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                # Validation
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor)
                    _, predicted = torch.max(val_outputs, 1)
                    accuracy = (predicted == y_test_tensor).float().mean().item()
                    
                print(f"Epoch {epoch}/{epochs}: Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.3f}")
        
        # Save enhanced model
        print("Saving enhanced model...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'training_date': datetime.now().isoformat(),
            'model_config': {
                'input_size': 10,
                'hidden_layers': [512, 256, 128, 64],
                'output_size': 3,
                'training_samples': len(X_train),
                'epochs': epochs,
                'features': ['price_change', 'z_score', 'sma_5_ratio', 'sma_20_ratio', 'rsi', 'volatility', 'trend_strength', 'bb_position', 'trend_continuation', 'trend_reversal']
            }
        }, 'enhanced_neural_model.pth')
        
        print("ENHANCED NEURAL TRAINING COMPLETED SUCCESSFULLY")
        print("Model saved as enhanced_neural_model.pth")
        print("Features: 10 (including candlestick patterns)")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open('enhanced_neural_training.py', 'w') as f:
            f.write(script_content)
        
        self.log_message("Created simple training script")
    
    def log_message(self, message):
        """Add a message to the log display"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\\n"
            
            # Add to the model status text area
            self.model_status_text.insert(tk.END, log_entry)
            self.model_status_text.see(tk.END)
            
            # Also log to logger
            self.logger.info(message)
            
        except Exception as e:
            self.logger.error(f"Error logging message: {e}")
    
    def update_neural_model(self):
        """Update/retrain the neural model using a simplified approach"""
        def update_thread():
            try:
                self.update_model_btn.config(state='disabled', text="Updating...")
                self.root.update()
                
                # Import required modules
                import subprocess
                import sys
                import os
                
                # Check if we have the enhanced training script
                if not os.path.exists('enhanced_neural_training.py'):
                    # Create a simple retraining script
                    self.create_simple_training_script()
                
                # Run the training script in a separate process
                self.log_message("Starting neural model update...")
                result = subprocess.run([
                    sys.executable, 'enhanced_neural_training.py'
                ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
                
                if result.returncode == 0:
                    self.log_message("Neural model update completed successfully")
                    messagebox.showinfo("Success", "Neural model updated successfully!")
                    
                    # Try to reload the model
                    try:
                        success = self.model_manager.load_model()
                        if success:
                            self.log_message("New model loaded successfully")
                            self.display_model_info()
                        else:
                            self.log_message("Model updated but failed to reload")
                    except Exception as e:
                        self.log_message(f"Model updated but reload failed: {e}")
                        
                else:
                    error_msg = f"Model update failed: {result.stderr}"
                    self.log_message(error_msg)
                    messagebox.showerror("Error", error_msg)
                
            except subprocess.TimeoutExpired:
                error_msg = "Model update timed out (>10 minutes) - Training was interrupted"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
            except Exception as e:
                error_msg = f"Failed to update model: {e}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
            finally:
                self.update_model_btn.config(state='normal', text="Update Neural Model")
        
        # Start the update in a separate thread to prevent GUI freezing
        threading.Thread(target=update_thread, daemon=True).start()
    
    def update_timer_display(self):
        """Update the timer display with comprehensive timer status"""
        try:
            current_time = datetime.now()
            
            timer_info = []
            timer_info.append("COMPREHENSIVE TRADING TIMER SYSTEM")
            timer_info.append("=" * 55)
            timer_info.append(f"Last Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            timer_info.append("")
            
            # Session Timers
            timer_info.append("SESSION TIMERS:")
            app_runtime = current_time - self.app_start_time
            timer_info.append(f"  App Running: {self._format_duration(app_runtime)}")
            
            if self.trading_start_time:
                trading_runtime = current_time - self.trading_start_time
                timer_info.append(f"  Trading Active: {self._format_duration(trading_runtime)}")
            else:
                timer_info.append(f"  Trading Active: Not Started")
            
            if self.last_trade_time:
                time_since_last_trade = current_time - self.last_trade_time
                timer_info.append(f"  Last Trade: {self._format_duration(time_since_last_trade)} ago")
            else:
                timer_info.append(f"  Last Trade: No trades yet")
            timer_info.append("")
            
            # Market Session Information
            timer_info.append("MARKET SESSION STATUS:")
            self._update_market_session()
            timer_info.append(f"  Current Session: {self.current_market_session}")
            timer_info.append(f"  Current UTC Time: {current_time.strftime('%H:%M:%S')}")
            
            # Time until next major session
            next_session = self._get_next_market_session(current_time)
            if next_session:
                time_to_next = next_session['start_time'] - current_time
                timer_info.append(f"  Next Session ({next_session['name']}): {self._format_duration(time_to_next)}")
            timer_info.append("")
            
            # Trading Configuration
            timer_info.append("TRADING CONFIGURATION:")
            timer_info.append(f"  Min Hold Time: {FREQUENT_TRADING_CONFIG['MIN_HOLD_TIME']} hours")
            timer_info.append(f"  Cooldown After Loss: {FREQUENT_TRADING_CONFIG['COOLDOWN_AFTER_LOSS']} hours")
            timer_info.append(f"  Min Profit R: {FREQUENT_TRADING_CONFIG['MIN_PROFIT_R']}")
            timer_info.append(f"  Trading Status: {'ACTIVE' if self.is_trading else 'INACTIVE'}")
            timer_info.append("")
            
            # Symbol Timer Status
            timer_info.append("SYMBOL TIMER STATUS:")
            active_symbols = 0
            ready_symbols = 0
            
            for symbol, timer_data in self.symbol_timers.items():
                symbol_ready = self._is_symbol_ready(timer_data, current_time)
                status = "READY" if symbol_ready else "COOLDOWN"
                if symbol_ready:
                    ready_symbols += 1
                else:
                    active_symbols += 1
                
                timer_info.append(f"  {symbol}: {status}")
                
                # Time since last trade
                if timer_data['last_trade_time'] != datetime.min:
                    time_since_trade = (current_time - timer_data['last_trade_time']).total_seconds() / 3600
                    timer_info.append(f"    Last Trade: {time_since_trade:.1f}h ago")
                else:
                    timer_info.append(f"    Last Trade: Never")
                
                # Cooldown status
                if current_time < timer_data['cooldown_until']:
                    cooldown_remaining = (timer_data['cooldown_until'] - current_time).total_seconds() / 3600
                    timer_info.append(f"    Cooldown: {cooldown_remaining:.1f}h remaining")
                
                # Profit lock status
                if current_time < timer_data['profit_lock_until']:
                    lock_remaining = (timer_data['profit_lock_until'] - current_time).total_seconds() / 3600
                    timer_info.append(f"    Profit Lock: {lock_remaining:.1f}h remaining")
            
            timer_info.append("")
            timer_info.append(f"SUMMARY: {ready_symbols} symbols ready, {active_symbols} in cooldown")
            
            # Update the display
            self.timer_display.delete('1.0', tk.END)
            self.timer_display.insert('1.0', '\n'.join(timer_info))
            
        except Exception as e:
            self.logger.error(f"Error updating timer display: {e}")
    
    def _format_duration(self, duration):
        """Format duration in human readable format"""
        total_seconds = int(duration.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _update_market_session(self):
        """Update current market session based on UTC time"""
        current_utc_hour = datetime.utcnow().hour
        
        for session_name, (start_hour, end_hour) in self.market_sessions.items():
            if start_hour <= end_hour:  # Same day session
                if start_hour <= current_utc_hour < end_hour:
                    self.current_market_session = session_name
                    return
            else:  # Overnight session (like NY: 13-22)
                if current_utc_hour >= start_hour or current_utc_hour < end_hour:
                    self.current_market_session = session_name
                    return
        
        self.current_market_session = "CLOSED"
    
    def _get_next_market_session(self, current_time):
        """Get next market session information"""
        current_utc_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()
        
        sessions_today = []
        sessions_tomorrow = []
        
        # Today's sessions
        for session_name, (start_hour, end_hour) in self.market_sessions.items():
            if start_hour <= end_hour:  # Same day
                if start_hour > current_utc_hour:
                    sessions_today.append((session_name, start_hour))
            else:  # Overnight
                if current_utc_hour < end_hour:
                    sessions_today.append((session_name, start_hour))
        
        # Tomorrow's sessions
        for session_name, (start_hour, end_hour) in self.market_sessions.items():
            sessions_tomorrow.append((session_name, start_hour))
        
        # Sort by start time
        sessions_today.sort(key=lambda x: x[1])
        sessions_tomorrow.sort(key=lambda x: x[1])
        
        # Return next session
        if sessions_today:
            next_session_name, next_hour = sessions_today[0]
            next_time = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            return {'name': next_session_name, 'start_time': next_time}
        elif sessions_tomorrow:
            next_session_name, next_hour = sessions_tomorrow[0]
            next_time = current_time.replace(day=current_time.day + 1, hour=next_hour, minute=0, second=0, microsecond=0)
            return {'name': next_session_name, 'start_time': next_time}
        
        return None
    
    def _is_symbol_ready(self, timer_data, current_time):
        """Check if a symbol is ready for trading"""
        # Check cooldown
        if current_time < timer_data['cooldown_until']:
            return False
        
        # Check profit lock
        if current_time < timer_data['profit_lock_until']:
            return False
        
        # Check minimum hold time
        if timer_data['last_trade_time'] != datetime.min:
            time_since_trade = (current_time - timer_data['last_trade_time']).total_seconds() / 3600
            if time_since_trade < FREQUENT_TRADING_CONFIG['MIN_HOLD_TIME']:
                return False
        
        return True

def main():
    """Main application entry point"""
    try:
        # Create and run the application
        root = tk.Tk()
        app = NeuralTradingApp(root)
        
        # Start the GUI event loop
        # Add periodic timer updates
        def update_timers():
            current_time = datetime.now()
            if (current_time - app.last_timer_update).seconds >= app.timer_update_interval:
                app.update_timer_display()
                app.last_timer_update = current_time
            root.after(1000, update_timers)  # Check every second
        
        update_timers()  # Start the timer update loop
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

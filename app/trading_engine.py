#!/usr/bin/env python3
"""
Neural Trading Engine
====================

Professional trading engine that integrates neural network predictions
with MT5 trading operations for automated forex trading.

Features:
- Neural network signal generation
- Automated trade execution
- Risk management
- Position monitoring
- Performance tracking
- Real-time trading loop
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

# Import app modules
from .mt5_connector import MT5Connector
from .model_manager import NeuralModelManager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from enhanced_tail_risk_protection import TailRiskProtector

class TradingSignal:
    """Trading signal data structure"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float, reason: str):
        self.symbol = symbol
        self.action = action  # 'BUY', 'SELL', 'HOLD'
        self.confidence = confidence
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.reason = reason
        self.timestamp = datetime.now()
        self.executed = False
        self.order_ticket = None

class Position:
    """Position tracking data structure"""
    
    def __init__(self, ticket: int, symbol: str, action: str, 
                 entry_price: float, stop_loss: float, take_profit: float,
                 position_size: float):
        self.ticket = ticket
        self.symbol = symbol
        self.action = action
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.open_time = datetime.now()
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.status = 'OPEN'  # OPEN, CLOSED, PARTIAL
        self.close_recorded = False

class TradingEngine:
    """Professional neural trading engine"""
    
    def __init__(self, mt5_connector: MT5Connector, model_manager: NeuralModelManager,
                 risk_per_trade: float = 0.015, confidence_threshold: float = 0.65,
                 trading_pairs: List[str] = None, max_concurrent_positions: int = 5):
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.mt5_connector = mt5_connector
        self.model_manager = model_manager
        
        # Enhanced tail risk protection
        self.tail_risk_protector = TailRiskProtector()
        
        # Advanced performance tracking
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from advanced_performance_tracking import AdvancedPerformanceTracker
        self.performance_tracker = AdvancedPerformanceTracker()
        
        # Trading parameters
        self.risk_per_trade = risk_per_trade  # 1.5% default
        self.confidence_threshold = confidence_threshold  # 65% default
        self.trading_pairs = trading_pairs or ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSD']
        self.max_concurrent_positions = max_concurrent_positions
        # Conservative startup mode: do not force entries unless explicitly enabled.
        # Immediate execution profile: prioritize taking valid model entries quickly.
        self.immediate_trade_mode = True
        self.profitability_first_mode = False
        self._startup_trade_done: set[str] = set()
        self.model_min_trade_score = 0.20  # EMERGENCY FIX: Was 0.33, too restrictive for profitability
        self.model_min_directional_gap = 0.005
        self.model_pattern_conflict_block = False
        self.mtf_alignment_enabled = False
        self.historical_signal_horizon = 16
        self.historical_min_samples = 30
        self.minimum_symbol_quality_winrate = 0.40  # EMERGENCY FIX: Was 0.50, too restrictive
        self.minimum_symbol_quality_samples = 20  # EMERGENCY FIX: Was 40, too restrictive
        self.minimum_symbol_profit_factor = 0.90  # EMERGENCY FIX: Was 1.05, blocking trades
        self.minimum_symbol_expectancy = 0.00001
        self.minimum_symbol_profitability_samples = 60
        self.live_min_trade_rate = 0.015
        self._symbol_live_profile: Dict[str, Dict[str, float]] = {}
        self._symbol_profile_skip_log_time: Dict[str, datetime] = {}
        self.market_closed_cooldown_seconds = 300
        self._symbol_trade_block_until: Dict[str, datetime] = {}
        self._symbol_market_closed_log_time: Dict[str, datetime] = {}
        self.symbol_entry_cooldown_seconds = 60
        self.max_new_trades_per_hour = 12
        self._symbol_last_entry_time: Dict[str, datetime] = {}
        self._new_trade_timestamps: List[datetime] = []
        # Tail-risk and intraday protection controls (left-tail avoidance).
        self.tail_risk_control_enabled = False
        self.tail_min_weekly_p10_return = 0.0
        self.tail_min_weekly_prob_positive = 0.60
        self.max_daily_loss_pct = 0.03
        self.max_intraday_drawdown_pct = 0.045
        self.loss_pause_until_next_day = True
        self.loss_streak_limit = 3
        self.loss_streak_cooldown_seconds = 1800
        self._symbol_loss_streak: Dict[str, int] = {}
        self._symbol_loss_block_until: Dict[str, datetime] = {}
        self._daily_risk_day: Optional[str] = None
        self._daily_start_equity = 0.0
        self._intraday_peak_equity = 0.0
        self._daily_loss_pause_until: Optional[datetime] = None
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.positions: Dict[int, Position] = {}
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        # Feature engineering cache
        self.feature_cache = {}
        self.last_update = {}
        
        # Timeframes for analysis
        self.timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        # Performance tracking
        self.start_time = None
        self.last_performance_update = None
        
        self.logger.info("Neural Trading Engine initialized")
    
    def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        # Validate prerequisites
        if not self.mt5_connector.is_connected():
            raise Exception("MT5 not connected")
        
        if not self.model_manager.is_model_loaded():
            raise Exception("Neural model not loaded")
        
        self.is_running = True
        self.start_time = datetime.now()
        self._startup_trade_done.clear()
        self._symbol_profile_skip_log_time.clear()
        self._symbol_trade_block_until.clear()
        self._symbol_market_closed_log_time.clear()
        self._symbol_last_entry_time.clear()
        self._new_trade_timestamps.clear()
        self._symbol_loss_streak.clear()
        self._symbol_loss_block_until.clear()
        self._daily_loss_pause_until = None
        self._refresh_account_risk_state(force_reset=True)
        self._rebuild_live_symbol_profile()
        if self.profitability_first_mode and self._symbol_live_profile:
            enabled_selected = [s for s in self.trading_pairs if self._is_symbol_live_enabled(s)]
            if enabled_selected:
                self.logger.info(
                    "Profitability-first mode active for selected symbols: "
                    + ", ".join(enabled_selected)
                )
            else:
                self.logger.warning(
                    "Profitability-first mode did not approve any selected symbols; "
                    "no trades will be executed until symbol quality improves"
                )
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Neural Trading Engine started")
    
    def stop(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        self.logger.info("Neural Trading Engine stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                self._refresh_account_risk_state()
                
                # Update positions
                self._update_positions()
                
                # Generate signals for trading pairs
                for symbol in self.trading_pairs:
                    if not self.is_running:
                        break
                    
                    try:
                        signal = self._generate_signal(symbol)
                        if signal:
                            self._process_signal(signal)
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update performance metrics
                if self.last_performance_update is None or \
                   (current_time - self.last_performance_update).seconds >= 60:
                    self._update_performance_metrics()
                    self.last_performance_update = current_time
                
                # Sleep before next iteration
                time.sleep(5)  # 5-second cycles
                
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {e}")
        finally:
            self.is_running = False

    def _get_symbol_profitability_gate(self, symbol: str) -> Tuple[bool, float, float, int]:
        """
        Return profitability gate status for a symbol.

        Returns:
            Tuple of:
            - is_weak: whether symbol should be treated as weak-profitability
            - avg_trade_return
            - profit_factor
            - sample_count
        """
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return False, 0.0, 0.0, 0

            profitability_map = metadata.get('symbol_profitability_validation', {})
            diagnostics_map = metadata.get('threshold_diagnostics', {})
            if not isinstance(profitability_map, dict):
                profitability_map = {}
            if not isinstance(diagnostics_map, dict):
                diagnostics_map = {}

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases: List[str] = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            avg_trade_return = 0.0
            profit_factor = 0.0
            profit_samples = 0
            selected_threshold = 0.0
            weekly_prob_positive = 0.5
            weekly_p10_return = -1.0

            for alias in aliases:
                stats = profitability_map.get(alias)
                if isinstance(stats, dict):
                    avg_trade_return = float(stats.get('avg_trade_return', 0.0) or 0.0)
                    profit_factor = float(stats.get('profit_factor', 0.0) or 0.0)
                    profit_samples = int(
                        round(float(stats.get('trade_count', stats.get('trades', 0.0)) or 0.0))
                    )
                    selected_threshold = float(stats.get('threshold', 0.0) or 0.0)
                    weekly_prob_positive = float(stats.get('weekly_prob_positive', weekly_prob_positive) or weekly_prob_positive)
                    weekly_p10_return = float(stats.get('weekly_p10_return', weekly_p10_return) or weekly_p10_return)
                diag = diagnostics_map.get(alias)
                if isinstance(diag, dict):
                    diag_expectancy = float(diag.get('expectancy', avg_trade_return) or avg_trade_return)
                    diag_pf = float(diag.get('profit_factor', profit_factor) or profit_factor)
                    diag_trades = int(round(float(diag.get('trade_count', profit_samples) or profit_samples)))
                    diag_threshold = float(diag.get('threshold', selected_threshold) or selected_threshold)
                    diag_weekly_prob = float(diag.get('weekly_prob_positive', weekly_prob_positive) or weekly_prob_positive)
                    diag_weekly_p10 = float(diag.get('weekly_p10_return', weekly_p10_return) or weekly_p10_return)
                    if profit_samples < self.minimum_symbol_profitability_samples:
                        avg_trade_return = diag_expectancy
                        profit_factor = diag_pf
                        profit_samples = diag_trades
                        selected_threshold = diag_threshold
                        weekly_prob_positive = diag_weekly_prob
                        weekly_p10_return = diag_weekly_p10
                if profit_samples > 0 or selected_threshold > 0:
                    break

            is_weak = bool(
                (
                    profit_samples >= self.minimum_symbol_profitability_samples
                    or selected_threshold >= 0.85
                )
                and (
                    avg_trade_return <= self.minimum_symbol_expectancy
                    or profit_factor < self.minimum_symbol_profit_factor
                )
            )
            if self.tail_risk_control_enabled and profit_samples > 0:
                if (
                    weekly_p10_return < self.tail_min_weekly_p10_return
                    or weekly_prob_positive < self.tail_min_weekly_prob_positive
                ):
                    is_weak = True
            return is_weak, float(avg_trade_return), float(profit_factor), int(profit_samples)
        except Exception:
            return False, 0.0, 0.0, 0

    def _resolve_symbol_trade_threshold(self, symbol: str) -> float:
        """Resolve symbol confidence threshold from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return float(self.confidence_threshold)

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases: List[str] = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            thresholds = metadata.get('symbol_thresholds', {})
            if isinstance(thresholds, dict):
                for alias in aliases:
                    value = thresholds.get(alias)
                    if value is not None:
                        try:
                            return float(value)
                        except Exception:
                            continue

            global_threshold = metadata.get('global_trade_threshold', self.confidence_threshold)
            return float(global_threshold)
        except Exception:
            return float(self.confidence_threshold)

    def _rebuild_live_symbol_profile(self) -> None:
        """Build symbol-level live trading profile from model validation metadata."""
        try:
            self._symbol_live_profile = {}
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return

            profitability_map = metadata.get('symbol_profitability_validation', {})
            diagnostics_map = metadata.get('threshold_diagnostics', {})
            configured_live_profile = metadata.get('symbol_live_profile', {})
            if not isinstance(profitability_map, dict):
                profitability_map = {}
            if not isinstance(diagnostics_map, dict):
                diagnostics_map = {}
            if not isinstance(configured_live_profile, dict):
                configured_live_profile = {}
            if not profitability_map and not diagnostics_map and not configured_live_profile:
                return

            profile_cfg = metadata.get('live_profile_config', {})
            if not isinstance(profile_cfg, dict):
                profile_cfg = {}
            min_samples = int(
                profile_cfg.get(
                    'min_samples',
                    self.minimum_symbol_profitability_samples,
                )
            )
            min_profit_factor = float(
                profile_cfg.get(
                    'min_profit_factor',
                    self.minimum_symbol_profit_factor,
                )
            )
            min_expectancy = float(
                profile_cfg.get(
                    'min_expectancy',
                    self.minimum_symbol_expectancy,
                )
            )
            min_trade_rate = float(
                profile_cfg.get(
                    'min_trade_rate',
                    self.live_min_trade_rate,
                )
            )
            min_weekly_prob_positive = float(
                profile_cfg.get(
                    'min_weekly_prob_positive',
                    self.tail_min_weekly_prob_positive,
                )
            )
            min_weekly_p10_return = float(
                profile_cfg.get(
                    'min_weekly_p10_return',
                    self.tail_min_weekly_p10_return,
                )
            )
            if self.tail_risk_control_enabled:
                min_weekly_prob_positive = max(min_weekly_prob_positive, self.tail_min_weekly_prob_positive)
                min_weekly_p10_return = max(min_weekly_p10_return, self.tail_min_weekly_p10_return)
            self.minimum_symbol_profitability_samples = max(1, min_samples)
            self.minimum_symbol_profit_factor = min_profit_factor
            self.minimum_symbol_expectancy = min_expectancy
            self.live_min_trade_rate = min_trade_rate

            all_symbols = sorted(
                set(profitability_map.keys())
                | set(diagnostics_map.keys())
                | set(configured_live_profile.keys())
            )
            for symbol in all_symbols:
                stats = profitability_map.get(symbol, {})
                diag = diagnostics_map.get(symbol, {})
                configured = configured_live_profile.get(symbol, {})
                if not isinstance(stats, dict):
                    stats = {}
                if not isinstance(diag, dict):
                    diag = {}
                if not isinstance(configured, dict):
                    configured = {}

                trade_count = int(
                    round(
                        float(
                            configured.get(
                                'trade_count',
                                stats.get(
                                    'trade_count',
                                    diag.get('trade_count', 0.0),
                                ),
                            ) or 0.0
                        )
                    )
                )
                samples = int(
                    round(
                        float(
                            configured.get(
                                'samples',
                                stats.get('samples', diag.get('samples', 0.0)),
                            ) or 0.0
                        )
                    )
                )
                expectancy = float(
                    configured.get(
                        'expectancy',
                        stats.get(
                            'avg_trade_return',
                            diag.get('expectancy', 0.0),
                        ),
                    ) or 0.0
                )
                profit_factor = float(
                    configured.get(
                        'profit_factor',
                        stats.get(
                            'profit_factor',
                            diag.get('profit_factor', 0.0),
                        ),
                    ) or 0.0
                )
                trade_rate = float(
                    configured.get(
                        'trade_rate',
                        stats.get('trade_rate', (trade_count / samples) if samples > 0 else 0.0),
                    ) or 0.0
                )
                weekly_prob_positive = float(
                    configured.get(
                        'weekly_prob_positive',
                        stats.get(
                            'weekly_prob_positive',
                            diag.get('weekly_prob_positive', 0.5),
                        ),
                    ) or 0.5
                )
                weekly_p10_return = float(
                    configured.get(
                        'weekly_p10_return',
                        stats.get(
                            'weekly_p10_return',
                            diag.get('weekly_p10_return', -1.0),
                        ),
                    ) or -1.0
                )
                metric_enabled = bool(
                    trade_count >= self.minimum_symbol_profitability_samples
                    and expectancy > self.minimum_symbol_expectancy
                    and profit_factor >= self.minimum_symbol_profit_factor
                    and trade_rate >= self.live_min_trade_rate
                    and weekly_prob_positive >= min_weekly_prob_positive
                    and weekly_p10_return >= min_weekly_p10_return
                )
                configured_enabled = configured.get('enabled', None)
                if configured_enabled is None:
                    enabled = metric_enabled
                else:
                    # Never allow stale metadata to force-enable weak symbols.
                    enabled = bool(configured_enabled) and metric_enabled
                if self.tail_risk_control_enabled and enabled:
                    if (
                        weekly_prob_positive < min_weekly_prob_positive
                        or weekly_p10_return < min_weekly_p10_return
                    ):
                        enabled = False

                configured_risk_multiplier = configured.get('risk_multiplier', None)
                if configured_risk_multiplier is not None:
                    risk_multiplier = float(np.clip(float(configured_risk_multiplier or 0.0), 0.0, 2.0))
                elif enabled:
                    pf_component = np.clip((profit_factor - self.minimum_symbol_profit_factor) / 0.35, 0.0, 1.0)
                    exp_denominator = max(self.minimum_symbol_expectancy * 4.0, 1e-8)
                    exp_component = np.clip((expectancy - self.minimum_symbol_expectancy) / exp_denominator, 0.0, 1.0)
                    sample_component = np.clip(trade_count / 250.0, 0.0, 1.0)
                    risk_multiplier = float(
                        np.clip(
                            0.55 + 0.30 * pf_component + 0.25 * exp_component + 0.20 * sample_component,
                            0.55,
                            1.30,
                        )
                    )
                else:
                    risk_multiplier = 0.0

                self._symbol_live_profile[str(symbol)] = {
                    'enabled': 1.0 if enabled else 0.0,
                    'risk_multiplier': risk_multiplier,
                    'expectancy': expectancy,
                    'profit_factor': profit_factor,
                    'trade_count': float(trade_count),
                    'trade_rate': trade_rate,
                    'weekly_prob_positive': weekly_prob_positive,
                    'weekly_p10_return': weekly_p10_return,
                }

            enabled_symbols = [s for s, v in self._symbol_live_profile.items() if v.get('enabled', 0.0) >= 1.0]
            if enabled_symbols:
                self.logger.info(
                    "Profitability-first live profile enabled symbols: "
                    + ", ".join(enabled_symbols)
                )
            else:
                self.logger.warning("Profitability-first live profile enabled no symbols")
        except Exception as e:
            self.logger.error(f"Failed to build symbol live profile: {e}")
            self._symbol_live_profile = {}

    def _resolve_symbol_live_profile(self, symbol: str) -> Optional[Dict[str, float]]:
        """Resolve per-symbol live profile entry using canonical aliases."""
        if not self._symbol_live_profile:
            return None
        normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
        aliases: List[str] = []
        if normalized:
            aliases.append(normalized)
            if normalized.endswith("USD") and len(normalized) > 3:
                aliases.append(normalized[:-3])
        for alias in aliases:
            entry = self._symbol_live_profile.get(alias)
            if isinstance(entry, dict):
                return entry
        return None

    def _is_symbol_live_enabled(self, symbol: str) -> bool:
        """Check if symbol is allowed for live entries by profitability profile."""
        if not self.profitability_first_mode:
            return True
        if not self._symbol_live_profile:
            return True
        entry = self._resolve_symbol_live_profile(symbol)
        if entry is None:
            return False
        return bool(entry.get('enabled', 0.0) >= 1.0)

    def _get_symbol_risk_multiplier(self, symbol: str) -> float:
        """Return symbol risk multiplier derived from profitability profile."""
        entry = self._resolve_symbol_live_profile(symbol)
        if entry is None:
            return 1.0
        if entry.get('enabled', 0.0) < 1.0:
            return 0.0
        return float(np.clip(float(entry.get('risk_multiplier', 1.0) or 1.0), 0.0, 2.0))

    def _log_symbol_profile_skip(self, symbol: str, reason: str, cooldown_seconds: int = 120) -> None:
        """Rate-limit repetitive profile skip logs per symbol."""
        now = datetime.now()
        key = str(symbol or "").upper()
        last = self._symbol_profile_skip_log_time.get(key)
        if last and (now - last).total_seconds() < cooldown_seconds:
            return
        self._symbol_profile_skip_log_time[key] = now
        self.logger.info(f"Skipping {symbol}: {reason}")

    def _refresh_account_risk_state(self, force_reset: bool = False) -> None:
        """Refresh intraday risk state and activate account-level pause when limits are hit."""
        try:
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return
            equity = float(account_info.get('equity', 0.0) or 0.0)
            balance = float(account_info.get('balance', 0.0) or 0.0)
            if equity <= 0.0:
                equity = balance
            if equity <= 0.0:
                return

            now = datetime.now()
            day_key = now.strftime("%Y-%m-%d")
            if force_reset or self._daily_risk_day != day_key or self._daily_start_equity <= 0.0:
                self._daily_risk_day = day_key
                self._daily_start_equity = equity
                self._intraday_peak_equity = equity
                self._daily_loss_pause_until = None
                self.performance_metrics['daily_pnl'] = 0.0
                return

            self._intraday_peak_equity = max(self._intraday_peak_equity, equity)
            daily_pnl = equity - self._daily_start_equity
            self.performance_metrics['daily_pnl'] = daily_pnl

            daily_loss_ratio = (
                (self._daily_start_equity - equity) / self._daily_start_equity
                if self._daily_start_equity > 0
                else 0.0
            )
            intraday_drawdown = (
                (self._intraday_peak_equity - equity) / self._intraday_peak_equity
                if self._intraday_peak_equity > 0
                else 0.0
            )
            self.performance_metrics['current_drawdown'] = max(
                float(self.performance_metrics.get('current_drawdown', 0.0) or 0.0),
                float(intraday_drawdown),
            )

            hit_daily_loss = daily_loss_ratio >= self.max_daily_loss_pct
            hit_intraday_dd = intraday_drawdown >= self.max_intraday_drawdown_pct
            if hit_daily_loss or hit_intraday_dd:
                if not self._daily_loss_pause_until or now >= self._daily_loss_pause_until:
                    if self.loss_pause_until_next_day:
                        self._daily_loss_pause_until = datetime(
                            year=now.year,
                            month=now.month,
                            day=now.day
                        ) + timedelta(days=1)
                    else:
                        self._daily_loss_pause_until = now + timedelta(hours=2)
                    self.logger.warning(
                        "Account risk pause activated: "
                        f"daily_loss={daily_loss_ratio:.2%}, "
                        f"intraday_drawdown={intraday_drawdown:.2%}, "
                        f"paused_until={self._daily_loss_pause_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to refresh account risk state: {e}")

    def _is_account_risk_paused(self) -> bool:
        """Return True if account-level risk pause is active."""
        if not self._daily_loss_pause_until:
            return False
        now = datetime.now()
        if now >= self._daily_loss_pause_until:
            self._daily_loss_pause_until = None
            return False
        return True

    def _register_closed_position(self, position: Position) -> None:
        """Track close events for symbol loss-streak cooldown."""
        try:
            if position.close_recorded:
                return
            position.close_recorded = True

            symbol_key = str(position.symbol or "").upper()
            pnl = float(position.unrealized_pnl or 0.0)
            if pnl < 0.0:
                streak = int(self._symbol_loss_streak.get(symbol_key, 0)) + 1
                self._symbol_loss_streak[symbol_key] = streak
                if streak >= self.loss_streak_limit:
                    block_until = datetime.now() + timedelta(seconds=self.loss_streak_cooldown_seconds)
                    self._symbol_loss_block_until[symbol_key] = block_until
                    self._symbol_loss_streak[symbol_key] = 0
                    self.logger.warning(
                        f"Loss-streak cooldown for {symbol_key}: "
                        f"{self.loss_streak_limit} consecutive losses, "
                        f"paused until {block_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            elif pnl > 0.0:
                self._symbol_loss_streak[symbol_key] = 0
        except Exception as e:
            self.logger.error(f"Failed to register closed position: {e}")
    
    def _generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate neural trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal object or None
        """
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return None

            # Get symbol info for trading
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Enhanced tail risk protection - analyze market conditions
            try:
                # Convert market_data to DataFrame for analysis
                if isinstance(market_data.get('rates'), list):
                    df = pd.DataFrame(market_data['rates'])
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('datetime', inplace=True)
                        market_conditions = self.tail_risk_protector.analyze_market_conditions(df)
                    else:
                        market_conditions = self.tail_risk_protector._get_default_risk_indicators()
                else:
                    market_conditions = self.tail_risk_protector._get_default_risk_indicators()
            except Exception as e:
                self.logger.warning(f"Failed to analyze market conditions for {symbol}: {e}")
                market_conditions = self.tail_risk_protector._get_default_risk_indicators()
            if not self._is_spread_acceptable(symbol_info):
                self.logger.debug(f"Spread too wide for {symbol}; skipping signal generation this cycle")
                return None
            if not self._is_symbol_live_enabled(symbol):
                self._log_symbol_profile_skip(symbol, "blocked by profitability-first live profile")
                return None
            risk_multiplier = self._get_symbol_risk_multiplier(symbol)
            if risk_multiplier <= 0.0:
                self._log_symbol_profile_skip(symbol, "non-positive live risk multiplier")
                return None

            profitability_weak, avg_trade_return, profit_factor, profit_samples = (
                self._get_symbol_profitability_gate(symbol)
            )
            quality_degraded_mode = False

            # If model validation quality is weak for this symbol, only allow
            # historical MTF entries with explicit positive edge.
            quality = self._get_model_symbol_quality(symbol)
            if quality:
                quality_win_rate, quality_samples = quality
                if (
                    quality_win_rate < self.minimum_symbol_quality_winrate
                    or quality_samples < self.minimum_symbol_quality_samples
                ):
                    self.logger.debug(
                        f"Model quality weak for {symbol} "
                        f"(dir_win={quality_win_rate:.3f}, samples={quality_samples}); "
                        "using historical MTF gate only"
                    )
                    if profitability_weak:
                        self.logger.debug(
                            f"Skipping {symbol}: weak directional quality and weak profitability "
                            f"(avg={avg_trade_return:.6f}, pf={profit_factor:.3f}, samples={profit_samples})"
                        )
                        return None
                    historical_signal = self._generate_historical_mtf_signal(symbol, symbol_info, market_data)
                    if historical_signal is not None:
                        return historical_signal

                    # Historical MTF fallback had no edge-valid setup.
                    # Allow neural path with stricter edge/confidence requirements.
                    quality_degraded_mode = True

            if profitability_weak and self.profitability_first_mode:
                self.logger.debug(
                    f"Model profitability weak for {symbol} "
                    f"(avg={avg_trade_return:.6f}, pf={profit_factor:.3f}, samples={profit_samples}); "
                    "skipping symbol this cycle"
                )
                return None
            
            # Extract features
            features = self._extract_features(symbol, market_data)
            if features is None:
                return None
            
            # Get neural prediction
            prediction = self.model_manager.predict(features, symbol=symbol)
            if not prediction:
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            action = str(prediction.get('action', 'HOLD')).upper()
            confidence = float(prediction.get('confidence', 0.0))
            model_threshold = float(prediction.get('trade_threshold', self.confidence_threshold))
            if self.profitability_first_mode or self.immediate_trade_mode:
                required_confidence = model_threshold
            else:
                required_confidence = max(self.confidence_threshold, model_threshold)
            should_trade = bool(prediction.get('should_trade', True))
            probabilities = prediction.get('probabilities', {}) if isinstance(prediction, dict) else {}
            buy_prob = float(probabilities.get('BUY', 0.0) or 0.0)
            sell_prob = float(probabilities.get('SELL', 0.0) or 0.0)
            directional_gap = abs(buy_prob - sell_prob)
            trade_score = float(prediction.get('trade_score', max(buy_prob, sell_prob)) or 0.0)

            if action not in ('BUY', 'SELL'):
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            if not should_trade or confidence < required_confidence:
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            min_trade_score = self.model_min_trade_score
            min_directional_gap = self.model_min_directional_gap

            if (
                trade_score < min_trade_score
                or directional_gap < min_directional_gap
            ):
                self.logger.debug(
                    f"Skipping {symbol}: weak model edge "
                    f"(score={trade_score:.3f}, gap={directional_gap:.3f})"
                )
                return self._generate_startup_fallback_signal(symbol, symbol_info, market_data)

            pattern_action, pattern_rule, pattern_streak = self._get_recent_candle_pattern_signal(
                market_data,
                timeframe='M15',
            )
            pattern_description = ""
            if pattern_action in ('BUY', 'SELL'):
                if pattern_rule == 'continuation_4plus':
                    pattern_description = (
                        f"M15 continuation: {pattern_streak} same-direction candles -> {pattern_action}"
                    )
                elif pattern_rule == 'reversal_3':
                    pattern_description = (
                        f"M15 reversal: 3 same-direction candles -> {pattern_action}"
                    )
                else:
                    pattern_description = f"M15 pattern signal -> {pattern_action}"
            pattern_confirmed = False
            if pattern_action in ('BUY', 'SELL'):
                if self.model_pattern_conflict_block and action != pattern_action:
                    self.logger.debug(
                        f"Skipping {symbol}: model={action} conflicts with M15 pattern={pattern_action} "
                        f"(rule={pattern_rule}, streak={pattern_streak})"
                    )
                    return None
                if action == pattern_action:
                    pattern_confirmed = True
                    confidence = float(np.clip(max(confidence, required_confidence), 0.0, 1.0))

            # Enforce multi-timeframe alignment for live model signals.
            if self.mtf_alignment_enabled:
                mtf_action, mtf_score, _ = self._get_current_mtf_action(market_data)
                if mtf_action in ('BUY', 'SELL') and mtf_action != action:
                    self.logger.debug(
                        f"Skipping {symbol} model signal {action}: MTF alignment is {mtf_action} (score {mtf_score})"
                    )
                    return None
            
            # Calculate trading parameters
            entry_price = symbol_info['ask'] if action == 'BUY' else symbol_info['bid']
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, action, entry_price, symbol_info, market_data
            )
            
            # Enhanced tail risk protection - validate signal
            if self.tail_risk_control_enabled:
                try:
                    # Get recent performance history for validation
                    recent_performance = [trade.get('pnl', 0.0) for trade in 
                                        self.performance_metrics.get('recent_trades', [])[-10:]]
                    
                    validation_result = self.tail_risk_protector.validate_trade_signal(
                        signal_confidence=confidence,
                        market_conditions=market_conditions,
                        recent_performance=recent_performance
                    )
                    
                    if not validation_result['approved']:
                        rejection_reasons = '; '.join(validation_result['rejection_reasons'])
                        self.logger.debug(
                            f"Tail risk protection rejected {symbol} {action} signal: {rejection_reasons}"
                        )
                        if not self.immediate_trade_mode:
                            return None
                    # Apply risk scaling from validation
                    risk_multiplier *= float(validation_result.get('position_size_multiplier', 1.0) or 1.0)
                    
                except Exception as e:
                    self.logger.warning(f"Tail risk validation failed for {symbol}: {e}")
                    # Continue without tail risk protection in case of error
            
            # Calculate position size with enhanced tail risk protection
            account_info = self.mt5_connector.get_account_info()
            base_position_size = self._calculate_position_size(
                symbol,
                entry_price,
                stop_loss,
                symbol_info,
                risk_multiplier=risk_multiplier,
            )
            
            if base_position_size <= 0:
                return None
            
            # Apply dynamic position sizing from tail risk protector
            try:
                enhanced_position_size = self.tail_risk_protector.calculate_dynamic_position_size(
                    base_lot_size=base_position_size,
                    confidence=confidence,
                    market_conditions=market_conditions,
                    account_balance=account_info.get('balance', 10000.0) if account_info else 10000.0
                )
                position_size = enhanced_position_size
            except Exception as e:
                self.logger.warning(f"Dynamic position sizing failed for {symbol}: {e}")
                position_size = base_position_size
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=(
                    (
                        f"Neural + pattern confirmation: {action} "
                        f"({confidence:.1%} confidence, threshold {required_confidence:.1%})"
                        if pattern_confirmed
                        else (
                            f"Neural prediction: {action} "
                            f"({confidence:.1%} confidence, threshold {required_confidence:.1%})"
                        )
                    )
                    + (f" | {pattern_description}" if pattern_description else "")
                )
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _generate_startup_fallback_signal(
        self,
        symbol: str,
        symbol_info: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate conservative fallback entries only when explicitly enabled.
        Fallback is restricted to historical MTF edge-validated setups.
        """
        try:
            if not self.immediate_trade_mode:
                return None
            if symbol in self._startup_trade_done:
                return None
            if not self._is_symbol_live_enabled(symbol):
                return None

            # Conservative fallback path: only historical MTF edge-validated entries.
            historical_signal = self._generate_historical_mtf_signal(symbol, symbol_info, market_data)
            if historical_signal is not None:
                return historical_signal

            return None
        except Exception as e:
            self.logger.error(f"Error generating startup fallback signal for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for a symbol"""
        try:
            market_data = {}
            
            # Get data for multiple timeframes
            for tf_name, tf_constant in self.timeframes.items():
                rates = self.mt5_connector.get_rates(symbol, tf_constant, 0, 100)
                if rates:
                    market_data[tf_name] = rates
                else:
                    self.logger.warning(f"No {tf_name} data for {symbol}")
                    return None
            
            # Get symbol info
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if symbol_info:
                market_data['symbol_info'] = symbol_info
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _prepare_ohlc_dataframe(self, rates: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Normalize MT5 rate dictionaries into a sorted OHLCV dataframe."""
        try:
            if not rates:
                return None
            df = pd.DataFrame(rates).copy()
            if df.empty:
                return None
            if 'time' not in df.columns:
                return None
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            df.drop_duplicates(subset='time', keep='last', inplace=True)
            df.set_index('time', inplace=True)
            for col in ('open', 'high', 'low', 'close', 'tick_volume'):
                if col not in df.columns:
                    if col == 'tick_volume':
                        df[col] = 0.0
                    else:
                        return None
            return df
        except Exception:
            return None

    def _get_recent_candle_pattern_signal(
        self,
        market_data: Dict[str, Any],
        timeframe: str = 'M15',
        lookback: int = 8,
    ) -> Tuple[str, str, int]:
        """
        Derive candle-pattern action from recent closed candles.

        Rule implemented:
        - 4+ candles same direction -> continuation
        - exactly 3 candles same direction -> reversal
        """
        try:
            rates = market_data.get(timeframe) or []
            if len(rates) < 6:
                return 'HOLD', '', 0

            # Exclude current forming candle; use only closed candles.
            closed = rates[:-1] if len(rates) > 1 else rates
            if len(closed) < 4:
                return 'HOLD', '', 0

            recent = closed[-max(4, int(lookback)):]
            directions: List[int] = []
            for candle in recent:
                open_price = float(candle.get('open', 0.0) or 0.0)
                close_price = float(candle.get('close', 0.0) or 0.0)
                if close_price > open_price:
                    directions.append(1)
                elif close_price < open_price:
                    directions.append(-1)
                else:
                    directions.append(0)

            if not directions or directions[-1] == 0:
                return 'HOLD', '', 0

            last_dir = directions[-1]
            streak = 1
            for idx in range(len(directions) - 2, -1, -1):
                if directions[idx] == last_dir:
                    streak += 1
                else:
                    break

            if streak >= 4:
                action = 'BUY' if last_dir > 0 else 'SELL'
                return action, 'continuation_4plus', int(streak)
            if streak == 3:
                action = 'SELL' if last_dir > 0 else 'BUY'
                return action, 'reversal_3', int(streak)
            return 'HOLD', '', int(streak)
        except Exception:
            return 'HOLD', '', 0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def _calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR series from OHLC dataframe."""
        prev_close = df['close'].shift(1)
        tr = pd.concat(
            [
                (df['high'] - df['low']).abs(),
                (df['high'] - prev_close).abs(),
                (df['low'] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _enrich_indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trainer-aligned indicators for runtime features and MTF logic."""
        out = df.copy()
        out['sma_5'] = out['close'].rolling(5).mean()
        out['sma_20'] = out['close'].rolling(20).mean()
        out['sma_50'] = out['close'].rolling(50).mean()
        out['ema_12'] = out['close'].ewm(span=12, adjust=False).mean()
        out['ema_26'] = out['close'].ewm(span=26, adjust=False).mean()
        out['rsi'] = self._calculate_rsi(out['close'])
        out['returns'] = out['close'].pct_change()
        out['volatility'] = out['returns'].rolling(20).std()
        out['atr_14'] = self._calculate_atr_series(out, period=14)
        bb_std = out['close'].rolling(20).std()
        out['bb_upper'] = out['sma_20'] + 2.0 * bb_std
        out['bb_lower'] = out['sma_20'] - 2.0 * bb_std
        out['volume_z'] = (
            (out['tick_volume'] - out['tick_volume'].rolling(20).mean())
            / (out['tick_volume'].rolling(20).std() + 1e-8)
        )
        candle_range = (out['high'] - out['low']).replace(0, np.nan)
        out['body_ratio'] = (out['close'] - out['open']) / (candle_range + 1e-8)
        return out.dropna()

    def _compute_timeframe_vote(self, row: pd.Series) -> int:
        """Compute simple directional vote from a timeframe indicator row."""
        vote = 0
        ema_fast = float(row.get('ema_12', 0.0))
        ema_slow = float(row.get('ema_26', 0.0))
        close = float(row.get('close', 0.0))
        sma_20 = float(row.get('sma_20', 0.0))
        rsi = float(row.get('rsi', 50.0))

        if ema_fast > ema_slow:
            vote += 1
        elif ema_fast < ema_slow:
            vote -= 1

        if close > sma_20:
            vote += 1
        elif close < sma_20:
            vote -= 1

        if rsi >= 55:
            vote += 1
        elif rsi <= 45:
            vote -= 1

        return vote

    def _get_current_mtf_action(self, market_data: Dict[str, Any]) -> Tuple[str, int, Dict[str, int]]:
        """Derive current multi-timeframe directional action from M5/M15/H1."""
        try:
            frames: Dict[str, pd.DataFrame] = {}
            for tf in ('M5', 'M15', 'H1'):
                rates = market_data.get(tf) or []
                frame = self._prepare_ohlc_dataframe(rates)
                if frame is None or len(frame) < 30:
                    return 'HOLD', 0, {}
                enriched = self._enrich_indicator_frame(frame)
                if enriched.empty:
                    return 'HOLD', 0, {}
                frames[tf] = enriched

            votes = {
                'M5': self._compute_timeframe_vote(frames['M5'].iloc[-1]),
                'M15': self._compute_timeframe_vote(frames['M15'].iloc[-1]),
                'H1': self._compute_timeframe_vote(frames['H1'].iloc[-1]),
            }
            total_vote = int(votes['M5'] + votes['M15'] + votes['H1'])

            # Require at least medium alignment and avoid H1 against the direction.
            if total_vote >= 3 and votes['M15'] > 0 and votes['H1'] >= 0:
                return 'BUY', total_vote, votes
            if total_vote <= -3 and votes['M15'] < 0 and votes['H1'] <= 0:
                return 'SELL', total_vote, votes
            return 'HOLD', total_vote, votes
        except Exception:
            return 'HOLD', 0, {}

    def _generate_historical_mtf_signal(
        self,
        symbol: str,
        symbol_info: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate startup signal from historical M15 pattern quality plus current MTF alignment.
        """
        try:
            mtf_action, mtf_score, votes = self._get_current_mtf_action(market_data)
            if mtf_action not in ('BUY', 'SELL'):
                return None

            m15_rates = market_data.get('M15') or []
            m15_frame = self._prepare_ohlc_dataframe(m15_rates)
            if m15_frame is None or len(m15_frame) < 80:
                return None
            m15 = self._enrich_indicator_frame(m15_frame)
            if len(m15) < 60:
                return None

            horizon = max(4, int(self.historical_signal_horizon))
            wins = 0
            trades = 0
            edge_sum = 0.0
            point = float(symbol_info.get("point", 0.0001) or 0.0001)
            live_spread_points = float(symbol_info.get("spread", 0.0) or 0.0)

            # Backtest the same M15-pattern direction logic on recent history.
            start_idx = 25
            end_idx = len(m15) - horizon
            for i in range(start_idx, end_idx):
                row = m15.iloc[i]
                vote = self._compute_timeframe_vote(row)

                if mtf_action == 'BUY' and vote < 2:
                    continue
                if mtf_action == 'SELL' and vote > -2:
                    continue

                entry = float(m15['close'].iloc[i])
                exit_price = float(m15['close'].iloc[i + horizon])
                future_return = (exit_price / (entry + 1e-12)) - 1.0

                pnl_directional = future_return if mtf_action == 'BUY' else -future_return
                spread_cost_ratio = (live_spread_points * point) / (entry + 1e-12)
                net_directional = pnl_directional - (spread_cost_ratio * 1.1)
                trades += 1
                edge_sum += net_directional
                if net_directional > 0:
                    wins += 1

            if trades < self.historical_min_samples:
                return None

            win_rate = wins / trades
            avg_edge = edge_sum / trades

            quality = self._get_model_symbol_quality(symbol)
            min_win_rate = 0.52
            min_avg_edge = 0.0
            if quality:
                quality_win_rate, quality_samples = quality
                if (
                    quality_win_rate < self.minimum_symbol_quality_winrate
                    or quality_samples < self.minimum_symbol_quality_samples
                ):
                    min_win_rate = 0.56
                    min_avg_edge = 0.00005

            # Require positive historical edge before forcing an immediate startup trade.
            if win_rate < min_win_rate or avg_edge <= min_avg_edge:
                return None

            startup_threshold = self._resolve_symbol_trade_threshold(symbol)
            confidence = float(
                np.clip(
                    0.56 + (win_rate - 0.5) * 1.4 + np.clip(avg_edge * 100.0, -0.04, 0.10),
                    max(startup_threshold + 0.01, 0.60),
                    0.92,
                )
            )

            entry_price = symbol_info['ask'] if mtf_action == 'BUY' else symbol_info['bid']
            stop_loss, take_profit = self._calculate_sl_tp(
                symbol, mtf_action, entry_price, symbol_info, market_data
            )
            position_size = self._calculate_position_size(
                symbol,
                entry_price,
                stop_loss,
                symbol_info,
                risk_multiplier=self._get_symbol_risk_multiplier(symbol),
            )
            if position_size <= 0:
                return None

            signal = TradingSignal(
                symbol=symbol,
                action=mtf_action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reason=(
                    f"Historical MTF startup entry ({mtf_action}, "
                    f"win={win_rate:.1%}, samples={trades}, votes={votes}, score={mtf_score})"
                ),
            )
            return signal
        except Exception as e:
            self.logger.error(f"Error building historical MTF startup signal for {symbol}: {e}")
            return None

    def _extract_features(self, symbol: str, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract runtime features aligned with trainer feature schema.
        Base feature order mirrors `simple_neural_trainer.py`.
        """
        try:
            if 'M15' not in market_data:
                return None

            m15_data = market_data['M15']
            if len(m15_data) < 80:
                return None

            df_raw = self._prepare_ohlc_dataframe(m15_data)
            if df_raw is None:
                return None
            df = self._enrich_indicator_frame(df_raw)
            if len(df) < 25:
                return None

            window = df.iloc[-20:]
            current = df.iloc[-1]
            prev = df.iloc[-2]

            current_price = float(current['close'])
            prev_price = float(prev['close'])
            price_std = float(window['close'].std())
            returns_std = float(window['returns'].std())

            if not np.isfinite(price_std) or price_std < 1e-12:
                price_std = 1e-8
            if not np.isfinite(returns_std) or returns_std < 1e-12:
                returns_std = 1e-8

            close_3 = float(df.iloc[-4]['close']) if len(df) >= 4 else prev_price
            close_6 = float(df.iloc[-7]['close']) if len(df) >= 7 else prev_price
            close_12 = float(df.iloc[-13]['close']) if len(df) >= 13 else prev_price

            atr_14 = float(current['atr_14']) if np.isfinite(current['atr_14']) else 0.0
            bb_upper = float(current['bb_upper']) if np.isfinite(current['bb_upper']) else current_price
            bb_lower = float(current['bb_lower']) if np.isfinite(current['bb_lower']) else current_price
            bb_width = max(bb_upper - bb_lower, 1e-8)
            bb_pos = (current_price - bb_lower) / bb_width
            volume_z = float(current['volume_z']) if np.isfinite(current['volume_z']) else 0.0
            body_ratio = float(current['body_ratio']) if np.isfinite(current['body_ratio']) else 0.0

            base_features = np.array(
                [
                    (current_price - prev_price) / (prev_price + 1e-12),
                    (current_price - float(window['close'].mean())) / (price_std + 1e-12),
                    float(current['sma_5']) / (current_price + 1e-12) - 1.0,
                    float(current['sma_20']) / (current_price + 1e-12) - 1.0,
                    float(current['rsi']) / 100.0,
                    returns_std * 100.0,
                    (current_price - close_3) / (close_3 + 1e-12),
                    (current_price - close_6) / (close_6 + 1e-12),
                    (current_price - close_12) / (close_12 + 1e-12),
                    float(current['sma_50']) / (current_price + 1e-12) - 1.0,
                    float(current['ema_12']) / (current_price + 1e-12) - 1.0,
                    float(current['ema_26']) / (current_price + 1e-12) - 1.0,
                    atr_14 / (current_price + 1e-12),
                    bb_pos,
                    volume_z,
                    body_ratio,
                ],
                dtype=np.float32,
            )

            symbol_features = self.model_manager.get_symbol_features(symbol)
            features = np.concatenate([base_features, symbol_features]).astype(np.float32)
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _estimate_atr(self, rates: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
        """Estimate ATR from rate dictionaries."""
        try:
            if not rates or len(rates) < period + 1:
                return None

            highs = np.array([float(r['high']) for r in rates], dtype=float)
            lows = np.array([float(r['low']) for r in rates], dtype=float)
            closes = np.array([float(r['close']) for r in rates], dtype=float)

            tr_values = []
            start_idx = max(1, len(rates) - period)
            for i in range(start_idx, len(rates)):
                high_low = highs[i] - lows[i]
                high_close = abs(highs[i] - closes[i - 1])
                low_close = abs(lows[i] - closes[i - 1])
                tr_values.append(max(high_low, high_close, low_close))

            if not tr_values:
                return None
            return float(np.mean(tr_values))
        except Exception:
            return None

    def _calculate_sl_tp(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        symbol_info: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for FX and non-FX symbols."""
        try:
            digits = int(symbol_info.get('digits', 5))
            point = float(symbol_info.get('point', 0.00001) or 0.00001)
            spread_points = float(symbol_info.get('spread', 0.0) or 0.0)
            spread_price = spread_points * point

            atr = None
            if market_data and 'M15' in market_data:
                atr = self._estimate_atr(market_data['M15'], period=14)

            # ATR-based distances are symbol-agnostic (works for BTCUSD too).
            if atr and atr > 0:
                base_distance = max(atr, spread_price * 4.0, point * 20.0)
            else:
                base_distance = max(spread_price * 30.0, point * 20.0, entry_price * 0.0005)

            sl_distance = base_distance * 1.2
            tp_distance = base_distance * 2.2

            if action == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance

            return round(stop_loss, digits), round(take_profit, digits)

        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return 0.0, 0.0
    
    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float,
                               symbol_info: Dict[str, Any], risk_multiplier: float = 1.0) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account info
            account_info = self.mt5_connector.get_account_info()
            if not account_info:
                return 0.0
            
            balance = float(account_info['balance'])
            risk_scale = float(np.clip(risk_multiplier, 0.0, 2.0))
            if risk_scale <= 0.0:
                return 0.0
            risk_amount = balance * self.risk_per_trade * risk_scale

            tick_size = float(symbol_info.get('trade_tick_size', 0.0) or 0.0)
            point = float(symbol_info.get('point', 0.0) or 0.0)
            if tick_size <= 0:
                tick_size = point if point > 0 else 1e-6

            tick_value = float(symbol_info.get('trade_tick_value', 0.0) or 0.0)
            if tick_value <= 0:
                contract_size = float(symbol_info.get('trade_contract_size', 1.0) or 1.0)
                tick_value = max(contract_size * tick_size, 1e-6)

            sl_distance = abs(entry_price - stop_loss)
            sl_ticks = sl_distance / tick_size

            if sl_ticks <= 0:
                return 0.0

            loss_per_lot = sl_ticks * tick_value
            if loss_per_lot <= 0:
                return 0.0

            # Calculate position size using symbol tick economics.
            position_size = risk_amount / loss_per_lot
            
            # Apply symbol constraints
            volume_min = float(symbol_info.get('volume_min', 0.01) or 0.01)
            volume_max = float(symbol_info.get('volume_max', 100.0) or 100.0)
            volume_step = float(symbol_info.get('volume_step', 0.01) or 0.01)
            
            # Round to step size
            position_size = round(position_size / volume_step) * volume_step
            
            # Ensure within limits
            position_size = max(volume_min, min(volume_max, position_size))
            
            return float(round(position_size, 2))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def _is_spread_acceptable(self, symbol_info: Dict[str, Any]) -> bool:
        """
        Validate quote quality before signal generation/execution.
        Uses tighter limits for FX and percentage limits for crypto/CFDs.
        """
        try:
            symbol = str(
                symbol_info.get('requested_name')
                or symbol_info.get('name')
                or ''
            ).upper()
            bid = float(symbol_info.get('bid', 0.0) or 0.0)
            ask = float(symbol_info.get('ask', 0.0) or 0.0)

            # If broker quote is not live, skip.
            if bid <= 0 or ask <= 0 or ask < bid:
                return False

            point = float(symbol_info.get('point', 0.0) or 0.0)
            spread_points = float(symbol_info.get('spread', 0.0) or 0.0)
            spread_price = spread_points * point if point > 0 else 0.0
            if spread_price <= 0:
                spread_price = ask - bid
            if spread_price <= 0:
                return False

            mid = (ask + bid) / 2.0
            if mid <= 0:
                return False
            spread_pct = spread_price / mid

            normalized = ''.join(ch for ch in symbol if ch.isalnum())
            fx_ccy = {"USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF"}
            is_crypto = any(token in normalized for token in ('BTC', 'ETH', 'XRP', 'LTC', 'SOL'))
            is_forex = (
                not is_crypto
                and len(normalized) >= 6
                and normalized[:3].isalpha()
                and normalized[3:6].isalpha()
                and normalized[:3] in fx_ccy
                and normalized[3:6] in fx_ccy
            )

            if is_forex:
                quote_ccy = normalized[3:6]
                pip_size = 0.01 if quote_ccy == 'JPY' else 0.0001
                spread_pips = spread_price / pip_size if pip_size > 0 else 999.0
                max_pips = 4.0 if quote_ccy == 'JPY' else 3.0
                if spread_pips > max_pips:
                    return False
                if spread_pct > 0.00035:
                    return False
                return True

            # Crypto/CFDs/metals: use percentage limits.
            if is_crypto:
                return spread_pct <= 0.0025
            return spread_pct <= 0.0015
        except Exception as e:
            self.logger.debug(f"Spread validation failed: {e}")
            return False

    def _get_model_symbol_quality(self, symbol: str) -> Optional[Tuple[float, int]]:
        """Return (directional_win_rate, directional_trade_count) from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return None

            symbol_validation = metadata.get('symbol_validation', {})
            if not isinstance(symbol_validation, dict) or not symbol_validation:
                return None

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            for alias in aliases:
                stats = symbol_validation.get(alias)
                if not isinstance(stats, dict):
                    continue
                win_rate = float(
                    stats.get('directional_win_rate', stats.get('win_rate', 0.0)) or 0.0
                )
                trade_count = int(
                    round(float(stats.get('directional_trade_count', stats.get('trade_count', 0.0)) or 0.0))
                )
                return win_rate, trade_count
            return None
        except Exception:
            return None

    def _get_model_symbol_profitability(self, symbol: str) -> Optional[Tuple[float, float, int]]:
        """Return (avg_trade_return, profit_factor, trade_count) from model metadata."""
        try:
            metadata = getattr(self.model_manager, 'model_metadata', {}) or {}
            if not isinstance(metadata, dict):
                return None

            profitability = metadata.get('symbol_profitability_validation', {})
            if not isinstance(profitability, dict) or not profitability:
                return None

            normalized = re.sub(r"[^A-Z0-9]", "", str(symbol or "").upper())
            aliases = []
            if normalized:
                aliases.append(normalized)
                if normalized.endswith("USD") and len(normalized) > 3:
                    aliases.append(normalized[:-3])

            for alias in aliases:
                stats = profitability.get(alias)
                if not isinstance(stats, dict):
                    continue

                avg_trade_return = float(stats.get('avg_trade_return', 0.0) or 0.0)
                profit_factor = float(stats.get('profit_factor', 0.0) or 0.0)
                trade_count = int(
                    round(float(stats.get('trade_count', stats.get('trades', 0.0)) or 0.0))
                )
                return avg_trade_return, profit_factor, trade_count

            return None
        except Exception:
            return None
    
    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            if signal.action not in ('BUY', 'SELL'):
                self.logger.info(
                    f"Skipping non-directional signal for {signal.symbol}: {signal.action}"
                )
                return

            # Check if we can take this trade
            if not self._can_trade(signal):
                return
            
            # Execute trade
            order_result = self._execute_trade(signal)
            if order_result:
                signal.executed = True
                signal.order_ticket = order_result.get('order')
                now = datetime.now()
                
                # Create position tracking
                position = Position(
                    ticket=order_result['order'],
                    symbol=signal.symbol,
                    action=signal.action,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size
                )
                
                self.positions[position.ticket] = position
                self.signals_history.append(signal)
                self._symbol_last_entry_time[signal.symbol.upper()] = now
                self._new_trade_timestamps.append(now)
                if "startup" in str(signal.reason).lower():
                    self._startup_trade_done.add(signal.symbol)
                
                # Record trade for performance tracking
                try:
                    trade_data = {
                        'symbol': signal.symbol,
                        'action': signal.action,
                        'entry_price': signal.entry_price,
                        'position_size': signal.position_size,
                        'confidence': signal.confidence,
                        'reason': signal.reason,
                        'pnl': 0,  # Will be updated when trade closes
                        'timestamp': datetime.now()
                    }
                    self.performance_tracker.record_trade(trade_data)
                except Exception as e:
                    self.logger.warning(f"Failed to record trade for performance tracking: {e}")
                
                self.logger.info(f"Trade executed: {signal.symbol} {signal.action} @ {signal.entry_price}")
                self.logger.info(f"SL: {signal.stop_loss}, TP: {signal.take_profit}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _can_trade(self, signal: TradingSignal) -> bool:
        """Check if we can execute the trade"""
        if signal.action not in ('BUY', 'SELL'):
            return False

        self._refresh_account_risk_state()
        if self._is_account_risk_paused():
            return False

        now = datetime.now()
        symbol_key = signal.symbol.upper()
        loss_block_until = self._symbol_loss_block_until.get(symbol_key)
        if loss_block_until:
            if now < loss_block_until:
                return False
            self._symbol_loss_block_until.pop(symbol_key, None)

        blocked_until = self._symbol_trade_block_until.get(symbol_key)
        if blocked_until:
            if now < blocked_until:
                return False
            # Cooldown expired.
            self._symbol_trade_block_until.pop(symbol_key, None)
            self._symbol_market_closed_log_time.pop(symbol_key, None)

        last_entry = self._symbol_last_entry_time.get(symbol_key)
        if last_entry and (now - last_entry).total_seconds() < self.symbol_entry_cooldown_seconds:
            return False

        cutoff = now - timedelta(hours=1)
        self._new_trade_timestamps = [ts for ts in self._new_trade_timestamps if ts >= cutoff]
        if len(self._new_trade_timestamps) >= self.max_new_trades_per_hour:
            return False

        # Check maximum concurrent positions
        if len(self.positions) >= self.max_concurrent_positions:
            return False
        
        # Check if we already have a position in this symbol
        for position in self.positions.values():
            if position.symbol == signal.symbol and position.status == 'OPEN':
                return False
        
        # Check confidence using symbol-level threshold when available.
        reason_text = str(getattr(signal, 'reason', '') or '').lower()
        symbol_threshold = self._resolve_symbol_trade_threshold(signal.symbol)
        if self.profitability_first_mode or self.immediate_trade_mode:
            required_confidence = symbol_threshold
        else:
            required_confidence = max(self.confidence_threshold, symbol_threshold)

        # Startup/historical fallback entries are generated from stricter
        # profitability and MTF gates, so do not over-block them with UI threshold.
        if "startup" in reason_text or "historical mtf" in reason_text:
            required_confidence = min(required_confidence, max(symbol_threshold, 0.58))

        if signal.confidence < required_confidence:
            return False
        
        # Additional risk checks can be added here
        
        return True
    
    def _execute_trade(self, signal: TradingSignal) -> Optional[Dict[str, Any]]:
        """Execute the actual trade"""
        try:
            # Prepare order request
            symbol_info = self.mt5_connector.get_symbol_info(signal.symbol)
            if not symbol_info:
                return None
            if not self._is_spread_acceptable(symbol_info):
                self.logger.debug(f"Execution skipped for {signal.symbol}: spread/quote quality failed")
                return None
            
            # Determine order type and price
            if signal.action == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                price = symbol_info['ask']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = symbol_info['bid']
            if price <= 0:
                self.logger.warning(f"Execution skipped for {signal.symbol}: invalid market price {price}")
                return None
            
            # Base order request; filling mode is selected with runtime fallback.
            base_order_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": signal.position_size,
                "type": order_type,
                "price": price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Neural-{signal.confidence:.1%}",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            invalid_fill_retcode = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)
            market_closed_retcode = getattr(mt5, 'TRADE_RETCODE_MARKET_CLOSED', 10018)
            fill_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
            tried = []

            for fill_mode in fill_modes:
                order_request = dict(base_order_request)
                order_request["type_filling"] = fill_mode
                tried.append(fill_mode)

                result = self.mt5_connector.send_order(order_request)
                if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                    return result

                if result and result.get('retcode') == market_closed_retcode:
                    symbol_key = signal.symbol.upper()
                    now = datetime.now()
                    blocked_until = now + timedelta(seconds=self.market_closed_cooldown_seconds)
                    self._symbol_trade_block_until[symbol_key] = blocked_until

                    last_log = self._symbol_market_closed_log_time.get(symbol_key)
                    if not last_log or (now - last_log).total_seconds() >= 60:
                        self._symbol_market_closed_log_time[symbol_key] = now
                        self.logger.warning(
                            f"Market closed for {signal.symbol}; pausing new attempts until "
                            f"{blocked_until.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    return None

                # Retry with another filling mode only when broker rejects filling policy.
                if not result or result.get('retcode') != invalid_fill_retcode:
                    self.logger.error(f"Trade execution failed: {result}")
                    return None

            self.logger.error(
                f"Trade execution failed after trying filling modes {tried}: invalid fill policy"
            )
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _update_positions(self):
        """Update position information and check for exits"""
        try:
            # Get current positions from MT5
            mt5_positions = self.mt5_connector.get_positions()
            
            # Update our position tracking
            for ticket, position in list(self.positions.items()):
                # Find position in MT5
                mt5_pos = None
                for pos in mt5_positions:
                    if pos['ticket'] == ticket:
                        mt5_pos = pos
                        break
                
                if mt5_pos:
                    # Update position data
                    position.current_price = mt5_pos['price_current']
                    position.unrealized_pnl = mt5_pos['profit']
                    
                    # Check exit conditions
                    if self._should_close_position(position, mt5_pos):
                        self._close_position(position)
                else:
                    # Position no longer exists in MT5, mark as closed
                    self._register_closed_position(position)
                    position.status = 'CLOSED'
                    self.logger.info(f"Position {ticket} closed")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _should_close_position(self, position: Position, mt5_pos: Dict[str, Any]) -> bool:
        """Determine if position should be closed"""
        try:
            current_price = mt5_pos['price_current']
            
            if position.action == 'BUY':
                # Check stop loss
                if current_price <= position.stop_loss:
                    return True
                # Check take profit
                if current_price >= position.take_profit:
                    return True
            else:  # SELL
                # Check stop loss
                if current_price >= position.stop_loss:
                    return True
                # Check take profit
                if current_price <= position.take_profit:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _close_position(self, position: Position):
        """Close a position"""
        try:
            # This would implement position closing logic
            # For now, just mark as closed
            self._register_closed_position(position)
            position.status = 'CLOSED'
            self.logger.info(f"Position {position.ticket} closed: {position.symbol} {position.action}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _update_performance_metrics(self):
        """Update trading performance metrics"""
        try:
            # Calculate metrics from positions and signals
            closed_positions = [p for p in self.positions.values() if p.status == 'CLOSED']
            
            if closed_positions:
                winning_trades = sum(1 for p in closed_positions if p.unrealized_pnl > 0)
                losing_trades = sum(1 for p in closed_positions if p.unrealized_pnl < 0)
                total_trades = len(closed_positions)
                
                self.performance_metrics.update({
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'total_pnl': sum(p.unrealized_pnl for p in closed_positions),
                    'current_drawdown': self._calculate_drawdown(closed_positions)
                })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _calculate_drawdown(self, positions: List[Position]) -> float:
        """Calculate current drawdown"""
        # Simplified drawdown calculation
        if not positions:
            return 0.0
        
        profits = [p.unrealized_pnl for p in positions]
        if not profits:
            return 0.0
        
        peak = max(profits)
        current = sum(profits)
        
        if peak <= 0:
            return 0.0
        
        return max(0, (peak - current) / peak)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get list of recent signals"""
        recent_signals = [s for s in self.signals_history[-20:]]  # Last 20 signals
        
        return [
            {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'reason': signal.reason,
                'executed': signal.executed
            }
            for signal in recent_signals
        ]
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get list of active positions"""
        active_positions = [p for p in self.positions.values() if p.status == 'OPEN']
        
        return [
            {
                'ticket': position.ticket,
                'symbol': position.symbol,
                'action': position.action,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'position_size': position.position_size,
                'unrealized_pnl': position.unrealized_pnl,
                'open_time': position.open_time.isoformat(),
                'status': position.status
            }
            for position in active_positions
        ]

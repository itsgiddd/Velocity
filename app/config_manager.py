#!/usr/bin/env python3
"""
Configuration Manager
====================

Professional configuration manager for the neural trading app.
Handles application settings, user preferences, and configuration persistence.

Features:
- Settings persistence and loading
- Default configuration values
- Configuration validation
- Environment-specific settings
- Backup and restore functionality
"""

import json
import logging

try:
    import yaml
except ImportError:
    yaml = None  # Will use JSON fallback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

class ConfigManager:
    """Professional configuration manager"""
    
    def __init__(self, config_dir: str = "config"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration file paths
        self.main_config_file = self.config_dir / "app_config.yaml"
        self.user_config_file = self.config_dir / "user_config.yaml"
        self.trading_config_file = self.config_dir / "trading_config.yaml"
        
        # Default configurations
        self.default_config = self._get_default_config()
        
        # Load existing configurations
        self.main_config = self._load_config(self.main_config_file, self.default_config['main'])
        self.user_config = self._load_config(self.user_config_file, self.default_config['user'])
        self.trading_config = self._load_config(self.trading_config_file, self.default_config['trading'])
        
        self.logger.info("Configuration Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default configuration values"""
        return {
            'main': {
                'app_name': 'ACI Trading System',
                'version': '2.0.0',
                'update_interval': 5,  # seconds
                'max_log_files': 10,
                'log_level': 'INFO',
                'auto_save_interval': 300,  # 5 minutes
                'theme': 'default',
                'window': {
                    'width': 1000,
                    'height': 700,
                    'resizable': True,
                    'center_on_screen': True
                }
            },
            'user': {
                'mt5': {
                    'server': 'auto',  # Auto-detect from MT5
                    'login': 'auto',  # Auto-detect from MT5
                    'password': 'auto',  # Use MT5 saved password
                    'auto_connect': False,
                    'connection_timeout': 30,
                    'retry_attempts': 3,
                    'auto_detect_credentials': True
                },
                'interface': {
                    'show_tooltips': True,
                    'auto_refresh_logs': True,
                    'confirm_trades': True,
                    'sound_notifications': False,
                    'show_performance_charts': True
                },
                'notifications': {
                    'trade_executions': True,
                    'errors': True,
                    'warnings': False,
                    'system_updates': False,
                    'email_notifications': False,
                    'email_address': ''
                }
            },
            'trading': {
                'general': {
                    'trading_mode': 'demo',  # demo, live
                    'default_risk_per_trade': 1.5,  # percentage
                    'default_confidence_threshold': 65,  # percentage
                    'max_concurrent_positions': 5,
                    'max_daily_trades': 50,
                    'max_daily_loss': 5.0,  # percentage
                    'auto_trading_enabled': False
                },
                'risk_management': {
                    'enable_stop_loss': True,
                    'enable_take_profit': True,
                    'enable_trailing_stop': False,
                    'enable_breakeven': False,
                    'position_sizing_method': 'fixed_risk',  # fixed_risk, fixed_size, volatility_adjusted
                    'correlation_filter': True,
                    'max_correlation': 0.7
                },
                'trading_pairs': {
                    'major_pairs': {
                        'EURUSD': {'enabled': True, 'max_risk': 2.0},
                        'GBPUSD': {'enabled': True, 'max_risk': 2.0},
                        'USDJPY': {'enabled': True, 'max_risk': 2.0},
                        'AUDUSD': {'enabled': True, 'max_risk': 2.0}
                    },
                    'minor_pairs': {
                        'USDCAD': {'enabled': False, 'max_risk': 1.5},
                        'NZDUSD': {'enabled': False, 'max_risk': 1.5},
                        'EURJPY': {'enabled': False, 'max_risk': 1.5},
                        'GBPJPY': {'enabled': False, 'max_risk': 1.5}
                    },
                    'crypto_pairs': {
                        'BTCUSD': {'enabled': True, 'max_risk': 1.0}
                    }
                },
                'neural_network': {
                    'model_path': 'models/neural_model.pth',
                    'auto_load_model': True,
                    'confidence_threshold': 65,
                    'min_trades_for_retrain': 100,
                    'retrain_interval_days': 30,
                    'feature_engineering': {
                        'use_multi_timeframe': True,
                        'technical_indicators': ['rsi', 'macd', 'bollinger', 'stochastic'],
                        'lookback_periods': [5, 20, 50],
                        'volatility_periods': [10, 20]
                    }
                },
                'execution': {
                    'order_type': 'market',  # market, limit, stop
                    'slippage_tolerance': 20,  # points
                    'max_spread': 5.0,  # pips
                    'trading_hours': {
                        'monday': {'start': '00:00', 'end': '23:59'},
                        'tuesday': {'start': '00:00', 'end': '23:59'},
                        'wednesday': {'start': '00:00', 'end': '23:59'},
                        'thursday': {'start': '00:00', 'end': '23:59'},
                        'friday': {'start': '00:00', 'end': '21:00'},
                        'saturday': {'start': '00:00', 'end': '00:00'},
                        'sunday': {'start': '21:00', 'end': '23:59'}
                    },
                    'avoid_news_times': True,
                    'news_buffer_minutes': 30
                },
                'monitoring': {
                    'enable_performance_tracking': True,
                    'enable_drawdown_monitoring': True,
                    'max_drawdown_percent': 10.0,
                    'enable_profit_target': True,
                    'daily_profit_target': 2.0,  # percentage
                    'weekly_profit_target': 10.0,  # percentage
                    'monthly_profit_target': 50.0  # percentage
                }
            }
        }
    
    def _load_config(self, config_file: Path, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.suffix.lower() == '.yaml' and yaml is not None:
                        config = yaml.safe_load(f)
                    elif config_file.suffix.lower() == '.yaml' and yaml is None:
                        # yaml not installed — skip loading
                        return default_config.copy()
                    else:
                        config = json.load(f)
                
                # Merge with defaults (recursive merge)
                return self._merge_config(default_config, config)
            else:
                return default_config.copy()
                
        except Exception as e:
            self.logger.error(f"Error loading config from {config_file}: {e}")
            return default_config.copy()
    
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config_type: str = 'all') -> bool:
        """
        Save configuration to files
        
        Args:
            config_type: 'main', 'user', 'trading', or 'all'
        """
        try:
            if config_type in ['main', 'all']:
                self._save_to_file(self.main_config_file, self.main_config)
            
            if config_type in ['user', 'all']:
                self._save_to_file(self.user_config_file, self.user_config)
            
            if config_type in ['trading', 'all']:
                self._save_to_file(self.trading_config_file, self.trading_config)
            
            self.logger.info(f"Configuration saved: {config_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def _save_to_file(self, config_file: Path, config: Dict[str, Any]) -> bool:
        """Save configuration to a specific file"""
        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.yaml' and yaml is not None:
                    yaml.dump(config, f, default_flow_style=False, indent=2, allow_unicode=True)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config to {config_file}: {e}")
            return False
    
    def get_config(self, config_type: str, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            config_type: 'main', 'user', or 'trading'
            key: Configuration key (supports dot notation like 'mt5.server')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Get the configuration dictionary
            if config_type == 'main':
                config = self.main_config
            elif config_type == 'user':
                config = self.user_config
            elif config_type == 'trading':
                config = self.trading_config
            else:
                return default
            
            # Handle key navigation with dot notation
            if key:
                keys = key.split('.')
                value = config
                
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                
                return value
            else:
                return config
                
        except Exception as e:
            self.logger.error(f"Error getting config {config_type}.{key}: {e}")
            return default
    
    def set_config(self, config_type: str, key: str, value: Any) -> bool:
        """
        Set configuration value
        
        Args:
            config_type: 'main', 'user', or 'trading'
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            # Get the configuration dictionary
            if config_type == 'main':
                config = self.main_config
            elif config_type == 'user':
                config = self.user_config
            elif config_type == 'trading':
                config = self.trading_config
            else:
                return False
            
            # Navigate to the parent dictionary
            keys = key.split('.')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
            
            self.logger.info(f"Config set: {config_type}.{key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting config {config_type}.{key}: {e}")
            return False
    
    def reset_config(self, config_type: str = 'all') -> bool:
        """
        Reset configuration to defaults
        
        Args:
            config_type: 'main', 'user', 'trading', or 'all'
        """
        try:
            if config_type in ['main', 'all']:
                self.main_config = self.default_config['main'].copy()
            
            if config_type in ['user', 'all']:
                self.user_config = self.default_config['user'].copy()
            
            if config_type in ['trading', 'all']:
                self.trading_config = self.default_config['trading'].copy()
            
            # Save the reset configuration
            self.save_config(config_type)
            
            self.logger.info(f"Configuration reset: {config_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
            return False
    
    def validate_config(self, config_type: str = 'all') -> Dict[str, List[str]]:
        """
        Validate configuration values
        
        Args:
            config_type: 'main', 'user', 'trading', or 'all'
            
        Returns:
            Dictionary of validation errors by section
        """
        errors = {}
        
        try:
            if config_type in ['main', 'all']:
                errors.update(self._validate_main_config())
            
            if config_type in ['user', 'all']:
                errors.update(self._validate_user_config())
            
            if config_type in ['trading', 'all']:
                errors.update(self._validate_trading_config())
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
        
        return errors
    
    def _validate_main_config(self) -> Dict[str, List[str]]:
        """Validate main configuration"""
        errors = {'main': []}
        
        # Validate update interval
        interval = self.get_config('main', 'update_interval')
        if not isinstance(interval, (int, float)) or interval <= 0 or interval > 60:
            errors['main'].append('update_interval must be between 0 and 60 seconds')
        
        # Validate window size
        width = self.get_config('main', 'window.width')
        height = self.get_config('main', 'window.height')
        if not isinstance(width, int) or width < 800:
            errors['main'].append('window.width must be at least 800')
        if not isinstance(height, int) or height < 600:
            errors['main'].append('window.height must be at least 600')
        
        return errors
    
    def _validate_user_config(self) -> Dict[str, List[str]]:
        """Validate user configuration"""
        errors = {'user': []}
        
        # Validate MT5 settings
        timeout = self.get_config('user', 'mt5.connection_timeout')
        if not isinstance(timeout, int) or timeout <= 0:
            errors['user'].append('mt5.connection_timeout must be a positive integer')
        
        retry_attempts = self.get_config('user', 'mt5.retry_attempts')
        if not isinstance(retry_attempts, int) or retry_attempts < 0 or retry_attempts > 10:
            errors['user'].append('mt5.retry_attempts must be between 0 and 10')
        
        return errors
    
    def _validate_trading_config(self) -> Dict[str, List[str]]:
        """Validate trading configuration"""
        errors = {'trading': []}
        
        # Validate risk settings
        risk_per_trade = self.get_config('trading', 'general.default_risk_per_trade')
        if not isinstance(risk_per_trade, (int, float)) or risk_per_trade <= 0 or risk_per_trade > 10:
            errors['trading'].append('default_risk_per_trade must be between 0 and 10 percent')
        
        confidence_threshold = self.get_config('trading', 'general.default_confidence_threshold')
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 50 or confidence_threshold > 95:
            errors['trading'].append('default_confidence_threshold must be between 50 and 95 percent')
        
        max_positions = self.get_config('trading', 'general.max_concurrent_positions')
        if not isinstance(max_positions, int) or max_positions <= 0 or max_positions > 20:
            errors['trading'].append('max_concurrent_positions must be between 1 and 20')
        
        return errors
    
    def backup_config(self, backup_name: str = None) -> str:
        """
        Create backup of current configuration
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            Path to backup file
        """
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"config_backup_{timestamp}.yaml"
            
            backup_file = self.config_dir / backup_name
            
            # Create backup with all configurations
            backup_config = {
                'backup_date': datetime.now().isoformat(),
                'main': self.main_config,
                'user': self.user_config,
                'trading': self.trading_config
            }
            
            self._save_to_file(backup_file, backup_config)
            
            self.logger.info(f"Configuration backed up to: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Configuration backup error: {e}")
            return ""
    
    def restore_config(self, backup_file: str) -> bool:
        """
        Restore configuration from backup
        
        Args:
            backup_file: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # Load backup configuration
            backup_config = self._load_config(backup_path, {})
            
            if 'main' in backup_config:
                self.main_config = backup_config['main']
            if 'user' in backup_config:
                self.user_config = backup_config['user']
            if 'trading' in backup_config:
                self.trading_config = backup_config['trading']
            
            # Save restored configuration
            self.save_config('all')
            
            self.logger.info(f"Configuration restored from: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration restore error: {e}")
            return False
    
    def export_config(self, export_file: str, include_sensitive: bool = False) -> bool:
        """
        Export configuration to file
        
        Args:
            export_file: Path to export file
            include_sensitive: Whether to include sensitive data (passwords, etc.)
            
        Returns:
            True if successful
        """
        try:
            export_path = Path(export_file)
            
            # Prepare export data
            export_data = {
                'export_date': datetime.now().isoformat(),
                'version': self.get_config('main', 'version'),
                'main': self.main_config,
                'user': self._sanitize_config(self.user_config, include_sensitive),
                'trading': self.trading_config
            }
            
            self._save_to_file(export_path, export_data)
            
            self.logger.info(f"Configuration exported to: {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration export error: {e}")
            return False
    
    def _sanitize_config(self, config: Dict[str, Any], include_sensitive: bool) -> Dict[str, Any]:
        """Remove or mask sensitive configuration values"""
        if include_sensitive:
            return config
        
        # Create a copy to avoid modifying original
        sanitized = config.copy()
        
        # Remove or mask sensitive fields
        sensitive_fields = ['password', 'secret', 'token', 'key']
        
        def sanitize_dict(d):
            if isinstance(d, dict):
                for key, value in list(d.items()):
                    if any(field in key.lower() for field in sensitive_fields):
                        d[key] = '***HIDDEN***'
                    elif isinstance(value, dict):
                        sanitize_dict(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                sanitize_dict(item)
        
        sanitize_dict(sanitized)
        return sanitized
    
    def get_available_backups(self) -> List[Dict[str, Any]]:
        """Get list of available configuration backups"""
        backups = []
        
        try:
            for backup_file in self.config_dir.glob("config_backup_*.yaml"):
                stat = backup_file.stat()
                
                backups.append({
                    'file_path': str(backup_file),
                    'file_name': backup_file.name,
                    'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'file_size_kb': round(stat.st_size / 1024, 1)
                })
            
            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x['created_date'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting backup list: {e}")
        
        return backups

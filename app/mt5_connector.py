#!/usr/bin/env python3
"""
MT5 Connector
=============

Professional MT5 connection handler for the neural trading app.
Handles connection, disconnection, and account information retrieval.

Features:
- Automatic MT5 connection and disconnection
- Account information management
- Symbol data retrieval
- Error handling and logging
- Connection status monitoring
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

class MT5Connector:
    """Professional MT5 connection handler"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._connected = False
        self._account_info = None
        self._last_connection_check = None
        self._symbol_cache: Dict[str, str] = {}
        self._symbol_warning_ts: Dict[str, datetime] = {}

    def _warn_symbol_issue(self, key: str, message: str, cooldown_seconds: int = 60) -> None:
        """Rate-limit noisy symbol warnings so logs stay readable."""
        now = datetime.now()
        last_time = self._symbol_warning_ts.get(key)
        if last_time and (now - last_time).total_seconds() < cooldown_seconds:
            return
        self._symbol_warning_ts[key] = now
        self.logger.warning(message)

    def _build_symbol_candidates(self, symbol: str) -> List[str]:
        """
        Build a ranked list of broker symbols for a requested symbol.
        Priority is exact and full-symbol matches (e.g., BTCUSD*) before base-only aliases (BTC).
        """
        requested = str(symbol or "").strip()
        if not requested:
            return []

        requested_upper = requested.upper()
        candidates: List[str] = []

        def add_candidate(name: Optional[str]) -> None:
            if not name:
                return
            if name not in candidates:
                candidates.append(name)

        # Always try exact request first.
        add_candidate(requested)

        symbols = mt5.symbols_get() or []
        names = [s.name for s in symbols if getattr(s, 'name', None)]
        if not names:
            return candidates

        upper_to_name = {name.upper(): name for name in names}
        add_candidate(upper_to_name.get(requested_upper))

        # Full symbol match first (e.g., BTCUSD, BTCUSDm, BTCUSD.i).
        full_matches = [name for name in names if requested_upper in name.upper()]
        full_matches.sort(key=lambda n: (0 if n.upper().startswith(requested_upper) else 1, len(n)))
        for match in full_matches:
            add_candidate(match)

        # For XXXUSD, keep USD variants ahead of base-only alias.
        if requested_upper.endswith("USD") and len(requested_upper) > 3:
            base = requested_upper[:-3]

            usd_variants = [
                name for name in names
                if name.upper().startswith(base) and "USD" in name.upper()
            ]
            usd_variants.sort(key=len)
            for match in usd_variants:
                add_candidate(match)

            add_candidate(upper_to_name.get(base))

            base_prefix = [name for name in names if name.upper().startswith(base)]
            base_prefix.sort(key=len)
            for match in base_prefix:
                add_candidate(match)

        # Cached symbols are useful, but should never outrank fresh full-symbol matches.
        cached = self._symbol_cache.get(requested_upper)
        if cached and mt5.symbol_info(cached) is not None:
            add_candidate(cached)
        elif cached:
            self._symbol_cache.pop(requested_upper, None)

        return candidates

    def _resolve_symbol(self, symbol: str, require_rates: bool = False, timeframe: int = mt5.TIMEFRAME_M5) -> str:
        """
        Resolve requested symbol to a broker-available symbol that can be selected.
        If `require_rates=True`, only return symbols with available rates for the timeframe.
        """
        requested = str(symbol or "").strip()
        if not requested:
            return requested

        upper_requested = requested.upper()
        for candidate in self._build_symbol_candidates(requested):
            # Ensure symbol is visible/selected before use.
            if not mt5.symbol_select(candidate, True):
                continue

            symbol_info = mt5.symbol_info(candidate)
            if symbol_info is None:
                continue

            if require_rates:
                rates = mt5.copy_rates_from_pos(candidate, timeframe, 0, 2)
                if rates is None or len(rates) == 0:
                    continue

            resolved = symbol_info.name if getattr(symbol_info, 'name', None) else candidate
            self._symbol_cache[upper_requested] = resolved
            return resolved

        return requested
    
    def connect(self, server: str = None, login: str = None, password: str = None) -> bool:
        """
        Connect to MT5
        
        Args:
            server: MT5 server address (optional - auto-detected if not provided)
            login: Account login (optional - auto-detected if not provided)
            password: Account password (optional - auto-detected if not provided)
            
        Returns:
            bool: True if connection successful
        """
        try:
            self.logger.info("Attempting to connect to MT5...")
            
            # Try automatic connection first (using saved MT5 credentials)
            if not mt5.initialize():
                error = mt5.last_error()
                self.logger.warning(f"MT5 automatic initialization failed: {error}")
                
                # Try with provided credentials
                if server and login and password:
                    self.logger.info("Trying with provided credentials...")
                    if not mt5.initialize(server=server, login=int(login), password=password):
                        error = mt5.last_error()
                        self.logger.error(f"MT5 connection with credentials failed: {error}")
                        return False
                else:
                    self.logger.error("MT5 connection failed - no credentials available")
                    return False
            
            # Get account info if already connected
            account_info = mt5.account_info()
            if account_info:
                # Refresh symbol resolution cache for current account/session.
                self._symbol_cache.clear()
                self._symbol_warning_ts.clear()
                self._account_info = {
                    'login': account_info.login,
                    'server': account_info.server,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'margin_free': account_info.margin_free,
                    'currency': account_info.currency,
                    'leverage': account_info.leverage
                }
                
                self._connected = True
                self._last_connection_check = datetime.now()
                
                self.logger.info(f"Connected to MT5 - Account: {account_info.login}")
                self.logger.info(f"Server: {account_info.server}")
                self.logger.info(f"Balance: ${account_info.balance:.2f}")
                
                return True
            else:
                self.logger.warning("No account information available")
                return False
                
        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from MT5
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self._connected:
                mt5.shutdown()
                self._connected = False
                self._account_info = None
                self._symbol_cache.clear()
                self._symbol_warning_ts.clear()
                self.logger.info("Disconnected from MT5")
                return True
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 disconnection error: {e}")
            return False
    
    def get_available_accounts(self) -> List[Dict[str, Any]]:
        """
        Get available MT5 accounts (login information)
        
        Returns:
            List[Dict]: Available account information
        """
        try:
            # Try to get login information from MT5 terminal
            if mt5.terminal_info():
                # Get account info if MT5 is running
                account_info = mt5.account_info()
                if account_info:
                    return [{
                        'login': account_info.login,
                        'server': account_info.server,
                        'name': account_info.name if hasattr(account_info, 'name') else 'Unknown',
                        'balance': account_info.balance,
                        'currency': account_info.currency,
                        'type': 'Demo' if 'demo' in account_info.server.lower() else 'Live'
                    }]
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve account info: {e}")
            return []
    
    def get_account_credentials(self) -> Optional[Dict[str, str]]:
        """
        Get MT5 account credentials (for display in GUI)
        
        Returns:
            Dict: Account credentials (server, login)
        """
        try:
            accounts = self.get_available_accounts()
            if accounts:
                account = accounts[0]  # Use the first available account
                return {
                    'server': account['server'],
                    'login': str(account['login'])
                }
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve credentials: {e}")
            return None
    
    def is_connected(self) -> bool:
        """
        Check if connected to MT5
        
        Returns:
            bool: True if connected
        """
        if not self._connected:
            return False
        
        # Verify connection is still active
        try:
            account_info = mt5.account_info()
            if account_info:
                self._last_connection_check = datetime.now()
                return True
            else:
                self._connected = False
                self.logger.warning("MT5 connection lost")
                return False
                
        except Exception as e:
            self._connected = False
            self.logger.error(f"Connection check failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get current account information
        
        Returns:
            Dict containing account info or None if not connected
        """
        try:
            if not self.is_connected():
                return None
            
            account_info = mt5.account_info()
            if account_info:
                return {
                    'login': account_info.login,
                    'server': account_info.server,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'margin_free': account_info.margin_free,
                    'currency': account_info.currency,
                    'leverage': account_info.leverage,
                    'trade_allowed': account_info.trade_allowed,
                    'trade_expert': account_info.trade_expert
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            
        Returns:
            Dict containing symbol info or None
        """
        try:
            if not self.is_connected():
                return None
            resolved_symbol = self._resolve_symbol(symbol, require_rates=False)

            # Ensure symbol is visible in Market Watch.
            if not mt5.symbol_select(resolved_symbol, True):
                self._warn_symbol_issue(
                    f"select_info:{symbol}",
                    f"Failed to select symbol {resolved_symbol} in MT5 for request {symbol}"
                )
                return None

            symbol_info = mt5.symbol_info(resolved_symbol)
            if symbol_info:
                return {
                    'name': symbol_info.name,
                    'requested_name': symbol,
                    'bid': symbol_info.bid,
                    'ask': symbol_info.ask,
                    'spread': symbol_info.spread,
                    'digits': symbol_info.digits,
                    'point': symbol_info.point,
                    'trade_tick_value': symbol_info.trade_tick_value,
                    'trade_tick_size': symbol_info.trade_tick_size,
                    'trade_contract_size': symbol_info.trade_contract_size,
                    'volume_min': symbol_info.volume_min,
                    'volume_max': symbol_info.volume_max,
                    'volume_step': symbol_info.volume_step,
                    'margin_initial': symbol_info.margin_initial,
                    'margin_maintenance': symbol_info.margin_maintenance,
                    'session_deals': symbol_info.session_deals,
                    'session_buy_orders': symbol_info.session_buy_orders,
                    'session_sell_orders': symbol_info.session_sell_orders,
                    'volume': symbol_info.volume,
                    'high': getattr(symbol_info, 'high', 0.0),
                    'low': getattr(symbol_info, 'low', 0.0)
                }

            self._warn_symbol_issue(
                f"info_missing:{symbol}",
                f"Symbol info unavailable for request {symbol} (resolved: {resolved_symbol})"
            )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_rates(self, symbol: str, timeframe: int, start_pos: int = 0, count: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Get price data for a symbol
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe constant
            start_pos: Starting position
            count: Number of bars to retrieve
            
        Returns:
            List of price bars or None
        """
        try:
            if not self.is_connected():
                return None
            resolved_symbol = self._resolve_symbol(symbol, require_rates=True, timeframe=timeframe)

            # Ensure symbol is visible in Market Watch.
            if not mt5.symbol_select(resolved_symbol, True):
                self._warn_symbol_issue(
                    f"select_rates:{symbol}",
                    f"Failed to select symbol {resolved_symbol} in MT5 for request {symbol}"
                )
                return None

            rates = mt5.copy_rates_from_pos(resolved_symbol, timeframe, start_pos, count)
            if rates is not None and len(rates) > 0:
                return [
                    {
                        'time': datetime.fromtimestamp(rate['time']),
                        'open': rate['open'],
                        'high': rate['high'],
                        'low': rate['low'],
                        'close': rate['close'],
                        'tick_volume': rate['tick_volume'],
                        'spread': rate['spread'],
                        'real_volume': rate['real_volume']
                    }
                    for rate in rates
                ]

            self._warn_symbol_issue(
                f"rates_missing:{symbol}:{timeframe}",
                f"No rates returned for {symbol} (resolved: {resolved_symbol}, timeframe: {timeframe})"
            )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting rates for {symbol}: {e}")
            return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions
        
        Returns:
            List of position dictionaries
        """
        try:
            if not self.is_connected():
                return []
            
            positions = mt5.positions_get()
            if positions:
                return [
                    {
                        'ticket': pos.ticket,
                        'time': datetime.fromtimestamp(pos.time),
                        'time_msc': pos.time_msc,
                        'type': pos.type,
                        'magic': pos.magic,
                        'identifier': pos.identifier,
                        'reason': pos.reason,
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'price_current': pos.price_current,
                        'swap': pos.swap,
                        'profit': pos.profit,
                        'symbol': pos.symbol,
                        'comment': pos.comment,
                        'external_id': pos.external_id
                    }
                    for pos in positions
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get current pending orders
        
        Returns:
            List of order dictionaries
        """
        try:
            if not self.is_connected():
                return []
            
            orders = mt5.orders_get()
            if orders:
                return [
                    {
                        'ticket': order.ticket,
                        'time_setup': datetime.fromtimestamp(order.time_setup),
                        'time_setup_msc': order.time_setup_msc,
                        'time_expiration': datetime.fromtimestamp(order.time_expiration),
                        'type': order.type,
                        'magic': order.magic,
                        'position_id': order.position_id,
                        'position_by_id': order.position_by_id,
                        'reason': order.reason,
                        'volume_initial': order.volume_initial,
                        'volume_current': order.volume_current,
                        'price_open': order.price_open,
                        'sl': order.sl,
                        'tp': order.tp,
                        'price_current': order.price_current,
                        'price_stoplimit': order.price_stoplimit,
                        'symbol': order.symbol,
                        'comment': order.comment,
                        'external_id': order.external_id
                    }
                    for order in orders
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    def send_order(self, order_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send a trading order
        
        Args:
            order_request: Dictionary containing order parameters
            
        Returns:
            Order result dictionary or None
        """
        try:
            if not self.is_connected():
                return None

            if 'symbol' in order_request:
                order_request = dict(order_request)
                resolved_symbol = self._resolve_symbol(order_request['symbol'], require_rates=False)
                if not mt5.symbol_select(resolved_symbol, True):
                    self._warn_symbol_issue(
                        f"select_order:{order_request['symbol']}",
                        f"Failed to select symbol {resolved_symbol} in MT5 for order request"
                    )
                    return None
                order_request['symbol'] = resolved_symbol

            result = mt5.order_send(order_request)
            if result:
                return {
                    'retcode': result.retcode,
                    'deal': result.deal,
                    'order': result.order,
                    'volume': result.volume,
                    'price': result.price,
                    'bid': result.bid,
                    'ask': result.ask,
                    'comment': result.comment,
                    'request_id': result.request_id,
                    'retcode_external': result.retcode_external
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return None
    
    def get_last_error(self) -> tuple:
        """
        Get last MT5 error
        
        Returns:
            Tuple of (error_code, error_message)
        """
        return mt5.last_error()
    
    def get_connection_time(self) -> Optional[datetime]:
        """
        Get last connection check time
        
        Returns:
            datetime of last check or None
        """
        return self._last_connection_check
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        if self._connected:
            self.disconnect()

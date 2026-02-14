"""Persistent trade journal backed by SQLite.

Records every closed trade with full model context so the agentic
orchestrator can detect performance degradation and trigger retraining.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    model_confidence: float
    model_action: str
    model_probabilities: str  # JSON string
    symbol_threshold: float
    action_mode: str
    entry_time: str
    exit_time: str
    position_size: float
    model_version: str
    stop_loss: float
    take_profit: float


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    pnl             REAL NOT NULL,
    pnl_pct         REAL NOT NULL,
    model_confidence REAL,
    model_action    TEXT,
    model_probabilities TEXT,
    symbol_threshold REAL,
    action_mode     TEXT,
    entry_time      TEXT,
    exit_time       TEXT,
    position_size   REAL,
    model_version   TEXT,
    stop_loss       REAL,
    take_profit     REAL
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades (exit_time)
"""

_INSERT = """
INSERT OR IGNORE INTO trades (
    trade_id, symbol, direction, entry_price, exit_price,
    pnl, pnl_pct, model_confidence, model_action,
    model_probabilities, symbol_threshold, action_mode,
    entry_time, exit_time, position_size, model_version,
    stop_loss, take_profit
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class TradeJournal:
    """Thread-safe, SQLite-backed trade journal."""

    def __init__(self, db_path: str = "trade_journal.db") -> None:
        self._db_path = str(Path(db_path).resolve())
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _row_to_record(row: tuple) -> TradeRecord:
        return TradeRecord(
            trade_id=row[0],
            symbol=row[1],
            direction=row[2],
            entry_price=row[3],
            exit_price=row[4],
            pnl=row[5],
            pnl_pct=row[6],
            model_confidence=row[7] or 0.0,
            model_action=row[8] or "",
            model_probabilities=row[9] or "{}",
            symbol_threshold=row[10] or 0.0,
            action_mode=row[11] or "normal",
            entry_time=row[12] or "",
            exit_time=row[13] or "",
            position_size=row[14] or 0.0,
            model_version=row[15] or "",
            stop_loss=row[16] or 0.0,
            take_profit=row[17] or 0.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(self, record: TradeRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    _INSERT,
                    (
                        record.trade_id,
                        record.symbol,
                        record.direction,
                        record.entry_price,
                        record.exit_price,
                        record.pnl,
                        record.pnl_pct,
                        record.model_confidence,
                        record.model_action,
                        record.model_probabilities,
                        record.symbol_threshold,
                        record.action_mode,
                        record.entry_time,
                        record.exit_time,
                        record.position_size,
                        record.model_version,
                        record.stop_loss,
                        record.take_profit,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_recent_trades(self, n: int = 100) -> List[TradeRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?", (n,)
                ).fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                conn.close()

    def get_trades_since(self, since: datetime) -> List[TradeRecord]:
        iso = since.isoformat()
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time ASC",
                    (iso,),
                ).fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                conn.close()

    def get_trades_for_symbol(self, symbol: str, limit: int = 200) -> List[TradeRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY exit_time DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
                return [self._row_to_record(r) for r in rows]
            finally:
                conn.close()

    def get_trade_count(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

    def get_trades_since_retrain(self, retrain_time: datetime) -> List[TradeRecord]:
        return self.get_trades_since(retrain_time)

    def get_rolling_stats(self, window: int = 50) -> Dict[str, float]:
        trades = self.get_recent_trades(window)
        if not trades:
            return {"win_rate": 0.0, "profit_factor": 0.0, "avg_pnl_pct": 0.0, "trade_count": 0}

        wins = sum(1 for t in trades if t.pnl > 0)
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / len(trades)

        return {
            "win_rate": wins / len(trades) if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 1e-12 else float("inf"),
            "avg_pnl_pct": avg_pnl_pct,
            "trade_count": len(trades),
        }

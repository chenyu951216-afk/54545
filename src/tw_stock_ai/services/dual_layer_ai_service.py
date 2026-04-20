from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from config.settings import settings

DB_PATH = str(Path(settings.data_dir) / settings.sqlite_db_name)


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exploration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT,
            setup_key TEXT,
            market_regime TEXT,
            result TEXT,
            confidence REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            pnl REAL DEFAULT 0,
            leverage REAL DEFAULT 0,
            margin REAL DEFAULT 0,
            fee REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS setup_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setup_key TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT,
            market_regime TEXT,
            pnl_net REAL DEFAULT 0,
            fee REAL DEFAULT 0,
            margin REAL DEFAULT 0,
            pnl_on_margin_pct REAL DEFAULT 0,
            effective_weight REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_setup_outcomes_key ON setup_outcomes(setup_key, created_at)")
    return conn


@dataclass
class EntryClassification:
    learning_tier: str
    setup_key: str
    effective_margin_floor: float
    desired_margin: float
    leverage: int
    leverage_cap: int


class DualLayerAIService:
    """
    Dual-layer AI:
    - exploration: small/noisy results only affect confidence
    - effective learning: only meaningful live results affect learning
    """

    def __init__(self) -> None:
        pass

    def _effective_margin_floor(self, equity: float, leverage_cap: int) -> float:
        equity_floor = max(float(equity or 0.0), 0.0) * settings.effective_learning_min_margin_ratio
        leverage_anchor = max(int(leverage_cap or 1), 1)
        micro_cap = max(settings.min_margin_floor_usdt, (settings.min_order_notional_usdt / leverage_anchor) * 1.35)
        return max(settings.min_margin_floor_usdt, min(equity_floor, micro_cap))

    def effective_pnl_floor(self, margin: float, fee: float = 0.0) -> float:
        margin_f = max(float(margin or 0.0), 0.0)
        fee_f = abs(float(fee or 0.0))
        dynamic_floor = max(0.08, margin_f * 0.12 + fee_f * 2.0)
        return min(float(settings.effective_learning_min_abs_pnl_usdt), dynamic_floor)

    def _setup_key(self, symbol: str, side: str, market_regime: str) -> str:
        return f"{symbol}:{side}:{market_regime}"

    def get_setup_edge(self, symbol: str, side: str, market_regime: str) -> Dict[str, Any]:
        setup_key = self._setup_key(symbol, side, market_regime or "unknown")
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT pnl_net, pnl_on_margin_pct, effective_weight
                FROM setup_outcomes
                WHERE setup_key=?
                ORDER BY id DESC
                LIMIT 40
                """,
                (setup_key,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "setup_key": setup_key,
                "sample_count": 0,
                "weighted_win_rate": 0.0,
                "avg_margin_return_pct": 0.0,
                "confidence_bias": 0.0,
                "threshold_shift": 0.0,
                "quality": "unknown",
                "recent_loss_streak": 0,
                "cooldown": False,
                "cooldown_reason": "",
            }

        weight_sum = 0.0
        win_weight = 0.0
        margin_return_sum = 0.0
        for pnl_net, pnl_on_margin_pct, effective_weight in rows:
            w = max(float(effective_weight or 0.0), 0.1)
            weight_sum += w
            if float(pnl_net or 0.0) > 0:
                win_weight += w
            margin_return_sum += float(pnl_on_margin_pct or 0.0) * w
        recent_loss_streak = 0
        for pnl_net, _, _ in rows:
            if float(pnl_net or 0.0) >= 0:
                break
            recent_loss_streak += 1
        weighted_win_rate = win_weight / max(weight_sum, 1e-9)
        avg_margin_return_pct = margin_return_sum / max(weight_sum, 1e-9)

        confidence_bias = 0.0
        threshold_shift = 0.0
        quality = "neutral"
        cooldown = False
        cooldown_reason = ""
        if len(rows) >= settings.setup_memory_min_samples:
            confidence_bias = max(
                -settings.setup_memory_confidence_bias_cap,
                min(
                    settings.setup_memory_confidence_bias_cap,
                    (weighted_win_rate - 0.5) * 0.08 + avg_margin_return_pct / 100.0 * 0.18,
                ),
            )
            threshold_shift = -confidence_bias * 0.65
            if weighted_win_rate >= 0.6 and avg_margin_return_pct > 0:
                quality = "proven"
            elif weighted_win_rate <= 0.4 and avg_margin_return_pct < 0:
                quality = "weak"
            if (
                recent_loss_streak >= settings.setup_memory_cooldown_loss_streak
                and weighted_win_rate <= settings.setup_memory_cooldown_max_win_rate
            ):
                cooldown = True
                cooldown_reason = "setup_memory_cooldown"
        return {
            "setup_key": setup_key,
            "sample_count": len(rows),
            "weighted_win_rate": round(weighted_win_rate, 6),
            "avg_margin_return_pct": round(avg_margin_return_pct, 6),
            "confidence_bias": round(confidence_bias, 6),
            "threshold_shift": round(threshold_shift, 6),
            "quality": quality,
            "recent_loss_streak": recent_loss_streak,
            "cooldown": cooldown,
            "cooldown_reason": cooldown_reason,
        }

    def get_confidence(self, symbol: str) -> float:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(confidence), 0.0) FROM exploration WHERE symbol=?",
                (symbol,),
            ).fetchone()
            return float(row[0] or 0.0)
        finally:
            conn.close()

    def classify_entry(
        self,
        *,
        symbol: str,
        side: str,
        market_regime: str,
        confidence: float,
        leverage: int,
        leverage_cap: int,
        desired_margin: float,
        equity: float,
    ) -> Dict[str, Any]:
        effective_margin_floor = self._effective_margin_floor(equity, leverage_cap)
        learning_tier = "effective" if (
            desired_margin >= effective_margin_floor and leverage >= max(1, int(leverage_cap * 0.6))
        ) else "exploration"

        setup_key = self._setup_key(symbol, side, market_regime or "unknown")
        return EntryClassification(
            learning_tier=learning_tier,
            setup_key=setup_key,
            effective_margin_floor=round(effective_margin_floor, 8),
            desired_margin=round(float(desired_margin or 0.0), 8),
            leverage=int(leverage or 1),
            leverage_cap=int(leverage_cap or 1),
        ).__dict__

    def record_exploration(
        self,
        symbol: str,
        side: str,
        market_regime: str,
        result: str,
        confidence_delta: float,
    ) -> None:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO exploration (symbol, side, setup_key, market_regime, result, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    side,
                    self._setup_key(symbol, side, market_regime or "unknown"),
                    market_regime or "unknown",
                    result,
                    float(confidence_delta or 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_learning_trade(
        self,
        symbol: str,
        pnl: float,
        leverage: float,
        margin: float,
        fee: float,
    ) -> None:
        if abs(float(pnl or 0.0)) < self.effective_pnl_floor(margin, fee):
            return
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO learning (symbol, pnl, leverage, margin, fee)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    float(pnl or 0.0),
                    float(leverage or 0.0),
                    float(margin or 0.0),
                    float(fee or 0.0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_setup_outcome(
        self,
        *,
        symbol: str,
        side: str,
        market_regime: str,
        pnl_net: float,
        fee: float,
        margin: float,
        count_in_learning: bool,
        effective_weight: float | None = None,
    ) -> None:
        margin_f = max(float(margin or 0.0), 0.0)
        pnl_f = float(pnl_net or 0.0)
        fee_f = float(fee or 0.0)
        applied_weight = float(
            effective_weight
            if effective_weight is not None
            else (settings.effective_learning_weight if count_in_learning else settings.exploration_learning_weight)
        )
        pnl_on_margin_pct = (pnl_f / margin_f * 100.0) if margin_f > 0 else 0.0
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO setup_outcomes (
                    setup_key, symbol, side, market_regime, pnl_net, fee, margin, pnl_on_margin_pct, effective_weight
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._setup_key(symbol, side, market_regime or "unknown"),
                    symbol,
                    side,
                    market_regime or "unknown",
                    pnl_f,
                    fee_f,
                    margin_f,
                    pnl_on_margin_pct,
                    applied_weight,
                ),
            )
            conn.commit()
        finally:
            conn.close()

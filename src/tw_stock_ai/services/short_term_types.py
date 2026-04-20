from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class IndicatorBundle:
    latest_close: float
    latest_volume: int
    ema20: float | None
    ema60: float | None
    rsi: float | None
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    adx: float | None
    atr: float | None
    average_volume_20d: float | None
    average_turnover_20d: float | None
    volume_ratio: float | None
    recent_high_20d: float | None
    recent_low_20d: float | None
    consolidation_range_ratio: float | None


@dataclass(slots=True)
class UniverseDecision:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass(slots=True)
class PatternDecision:
    breakout_detected: bool
    consolidation_detected: bool
    pattern_label: str
    confidence: float
    metrics: dict = field(default_factory=dict)


@dataclass(slots=True)
class RiskPlan:
    entry_zone_low: float | None
    entry_zone_high: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    risk_reward_ratio: float | None
    holding_days_min: int | None
    holding_days_max: int | None
    risk_flags: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


@dataclass(slots=True)
class CandidateEvaluation:
    symbol: str
    symbol_name: str | None
    status: str
    as_of_date: date | None
    overall_score: float | None
    sub_scores: dict
    evidence: dict
    entry_zone_low: float | None
    entry_zone_high: float | None
    stop_loss: float | None
    take_profit: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    risk_reward_ratio: float | None
    holding_days_min: int | None
    holding_days_max: int | None
    risk_flags: dict

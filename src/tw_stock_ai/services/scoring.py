from __future__ import annotations

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.models import FundamentalSnapshot, NewsItem, PriceBar, RevenueSnapshot
from tw_stock_ai.services.indicators import calculate_indicator_bundle
from tw_stock_ai.services.patterns import detect_patterns
from tw_stock_ai.services.risk_engine import build_risk_plan
from tw_stock_ai.services.short_term_types import CandidateEvaluation, IndicatorBundle, PatternDecision, UniverseDecision
from tw_stock_ai.services.universe import apply_universe_filter
from tw_stock_ai.services.value_engine import build_value_payload


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _build_sub_scores(
    indicators: IndicatorBundle,
    pattern: PatternDecision,
    universe: UniverseDecision,
    risk_ratio: float | None,
    settings: Settings,
) -> dict[str, float]:
    trend_score = 0.0
    if indicators.ema20 is not None and indicators.ema60 is not None:
        if indicators.latest_close > indicators.ema20 > indicators.ema60:
            trend_score = 95.0
        elif indicators.latest_close > indicators.ema20 and indicators.ema20 >= indicators.ema60:
            trend_score = 80.0
        elif indicators.latest_close > indicators.ema60:
            trend_score = 60.0
        else:
            trend_score = 25.0

    momentum_score = 0.0
    if indicators.rsi is not None:
        if 52 <= indicators.rsi <= 68:
            momentum_score += 55.0
        elif 45 <= indicators.rsi <= 72:
            momentum_score += 40.0
        else:
            momentum_score += 20.0
    if (indicators.macd_histogram or 0.0) > 0:
        momentum_score += 45.0
    momentum_score = min(momentum_score, 100.0)

    volume_score = _clip((indicators.volume_ratio or 0.0) / settings.volume_surge_ratio_threshold * 100)
    pattern_score = max(pattern.confidence, 20.0 if universe.eligible else 0.0)
    strength_score = _clip(((indicators.adx or 0.0) / max(settings.adx_trend_threshold, 1.0)) * 60 + 20)
    risk_score = _clip((risk_ratio or 0.0) / max(settings.risk_min_reward_risk_ratio, 0.1) * 70)

    return {
        "trend_score": round(trend_score, 2),
        "momentum_score": round(momentum_score, 2),
        "volume_score": round(volume_score, 2),
        "pattern_score": round(pattern_score, 2),
        "strength_score": round(strength_score, 2),
        "risk_score": round(risk_score, 2),
    }


def _build_overall_score(sub_scores: dict[str, float], settings: Settings) -> float:
    return round(
        sub_scores["trend_score"] * settings.scoring_weight_trend
        + sub_scores["momentum_score"] * settings.scoring_weight_momentum
        + sub_scores["volume_score"] * settings.scoring_weight_volume
        + sub_scores["pattern_score"] * settings.scoring_weight_pattern
        + sub_scores["strength_score"] * settings.scoring_weight_strength
        + sub_scores["risk_score"] * settings.scoring_weight_risk,
        2,
    )


def build_candidate_payload(
    symbol: str,
    bars: list[PriceBar],
    fundamental: FundamentalSnapshot | None,
    min_bars: int,
    *,
    revenue_snapshot: RevenueSnapshot | None = None,
    news_items: list[NewsItem] | None = None,
    symbol_name: str | None = None,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    bars = sorted(bars, key=lambda item: item.trade_date)
    latest = bars[-1] if bars else None

    if latest is None or len(bars) < min_bars:
        return {
            "symbol": symbol,
            "symbol_name": symbol_name,
            "status": "missing_data",
            "overall_score": None,
            "sub_scores": {
                "trend_score": 0.0,
                "momentum_score": 0.0,
                "volume_score": 0.0,
                "pattern_score": 0.0,
                "strength_score": 0.0,
                "risk_score": 0.0,
            },
            "evidence": {
                "reason": "insufficient_price_bars",
                "bar_count": len(bars),
                "required_bar_count": min_bars,
                "latest_trade_date": latest.trade_date.isoformat() if latest else None,
            },
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop_loss": None,
            "take_profit": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "risk_reward_ratio": None,
            "holding_days_min": settings.short_term_holding_days_min,
            "holding_days_max": settings.short_term_holding_days_max,
            "risk_flags": {"reasons": ["insufficient_price_bars"]},
            **build_value_payload(
                fundamental=fundamental,
                revenue_snapshot=revenue_snapshot,
                news_items=news_items,
                settings=settings,
            ),
        }

    indicators = calculate_indicator_bundle(bars, settings)
    universe = apply_universe_filter(bars, indicators, settings)
    pattern = detect_patterns(bars, indicators, settings)
    risk_plan = build_risk_plan(indicators, pattern, settings)

    if indicators is None:
        status = "missing_data"
        sub_scores = {
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volume_score": 0.0,
            "pattern_score": 0.0,
            "strength_score": 0.0,
            "risk_score": 0.0,
        }
        overall_score = None
    else:
        sub_scores = _build_sub_scores(indicators, pattern, universe, risk_plan.risk_reward_ratio, settings)
        overall_score = _build_overall_score(sub_scores, settings)
        status = "ready"
        if not universe.eligible:
            status = "filtered_out"
        if risk_plan.risk_reward_ratio is None or risk_plan.risk_reward_ratio < settings.risk_min_reward_risk_ratio:
            status = "filtered_out"

    evidence = {
        "latest_trade_date": latest.trade_date.isoformat(),
        "latest_close": float(latest.close),
        "bar_count": len(bars),
        "source_name": latest.source_name,
        "source_url": latest.source_url,
        "fetched_at": latest.fetched_at.isoformat(),
        "universe": universe.metrics,
        "pattern": {"label": pattern.pattern_label, **pattern.metrics},
        "ema20": round(indicators.ema20, 4) if indicators and indicators.ema20 is not None else None,
        "ema60": round(indicators.ema60, 4) if indicators and indicators.ema60 is not None else None,
        "rsi": round(indicators.rsi, 2) if indicators and indicators.rsi is not None else None,
        "macd_line": round(indicators.macd_line, 4) if indicators and indicators.macd_line is not None else None,
        "macd_signal": round(indicators.macd_signal, 4) if indicators and indicators.macd_signal is not None else None,
        "macd_histogram": round(indicators.macd_histogram, 4) if indicators and indicators.macd_histogram is not None else None,
        "adx": round(indicators.adx, 2) if indicators and indicators.adx is not None else None,
        "atr": round(indicators.atr, 4) if indicators and indicators.atr is not None else None,
        "average_volume_20d": round(indicators.average_volume_20d, 2)
        if indicators and indicators.average_volume_20d is not None
        else None,
        "average_turnover_20d": round(indicators.average_turnover_20d, 2)
        if indicators and indicators.average_turnover_20d is not None
        else None,
        "volume_ratio": round(indicators.volume_ratio, 2) if indicators and indicators.volume_ratio is not None else None,
        "risk": risk_plan.metrics,
    }

    value_payload = build_value_payload(
        fundamental=fundamental,
        revenue_snapshot=revenue_snapshot,
        news_items=news_items,
        settings=settings,
    )

    risk_reasons = list(risk_plan.risk_flags.get("reasons", []))
    if symbol_name is None:
        risk_reasons.append("symbol_name_unavailable")
    if not universe.eligible:
        risk_reasons.extend(universe.reasons)

    candidate = CandidateEvaluation(
        symbol=symbol,
        symbol_name=symbol_name,
        status=status,
        as_of_date=latest.trade_date,
        overall_score=overall_score,
        sub_scores=sub_scores,
        evidence=evidence,
        entry_zone_low=risk_plan.entry_zone_low,
        entry_zone_high=risk_plan.entry_zone_high,
        stop_loss=risk_plan.stop_loss,
        take_profit=risk_plan.take_profit_1,
        take_profit_1=risk_plan.take_profit_1,
        take_profit_2=risk_plan.take_profit_2,
        risk_reward_ratio=risk_plan.risk_reward_ratio,
        holding_days_min=risk_plan.holding_days_min,
        holding_days_max=risk_plan.holding_days_max,
        risk_flags={"reasons": sorted(set(risk_reasons))},
    )

    return {
        "symbol": candidate.symbol,
        "symbol_name": candidate.symbol_name,
        "status": candidate.status,
        "overall_score": candidate.overall_score,
        "sub_scores": candidate.sub_scores,
        "evidence": candidate.evidence,
        "entry_zone_low": candidate.entry_zone_low,
        "entry_zone_high": candidate.entry_zone_high,
        "stop_loss": candidate.stop_loss,
        "take_profit": candidate.take_profit,
        "take_profit_1": candidate.take_profit_1,
        "take_profit_2": candidate.take_profit_2,
        "risk_reward_ratio": candidate.risk_reward_ratio,
        "holding_days_min": candidate.holding_days_min,
        "holding_days_max": candidate.holding_days_max,
        "risk_flags": candidate.risk_flags,
        **value_payload,
    }

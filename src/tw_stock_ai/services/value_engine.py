from __future__ import annotations

from datetime import datetime, timedelta

from tw_stock_ai.config import Settings
from tw_stock_ai.models import FundamentalSnapshot, NewsItem, RevenueSnapshot


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_higher_better(value: float | None, target: float) -> float | None:
    if value is None:
        return None
    if target == 0:
        return 50.0 if value >= 0 else 0.0
    return _clip((value / target) * 70 + 20)


def _score_lower_better(value: float | None, target: float) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 100.0
    return _clip((target / value) * 70 + 20)


def _average(scores: list[float | None]) -> float | None:
    available = [score for score in scores if score is not None]
    if not available:
        return None
    return round(sum(available) / len(available), 2)


def _keyword_list(settings: Settings) -> list[str]:
    return [item.strip() for item in settings.treasure_catalyst_keywords.split(",") if item.strip()]


def build_value_payload(
    *,
    fundamental: FundamentalSnapshot | None,
    revenue_snapshot: RevenueSnapshot | None,
    news_items: list[NewsItem] | None,
    settings: Settings,
) -> dict:
    news_items = news_items or []
    matched_titles: list[dict] = []
    catalyst_keywords = _keyword_list(settings)
    recent_cutoff = datetime.now().astimezone() - timedelta(days=settings.treasure_news_lookback_days)

    for item in news_items[: settings.treasure_recent_news_limit]:
        title = item.title or ""
        matched_keywords = [keyword for keyword in catalyst_keywords if keyword.lower() in title.lower()]
        if matched_keywords:
            matched_titles.append(
                {
                    "title": title,
                    "published_at": item.published_at.isoformat(),
                    "source_name": item.source_name,
                    "source_url": item.source_url,
                    "matched_keywords": matched_keywords,
                }
            )

    growth_score = _average(
        [
            _score_higher_better(revenue_snapshot.revenue_yoy if revenue_snapshot else None, settings.treasure_revenue_yoy_good),
            _score_higher_better(revenue_snapshot.revenue_mom if revenue_snapshot else None, settings.treasure_revenue_mom_good),
            _score_higher_better(fundamental.eps if fundamental else None, settings.treasure_eps_good),
        ]
    )
    quality_score = _average(
        [
            _score_higher_better(fundamental.roe if fundamental else None, settings.treasure_roe_good),
            _score_higher_better(fundamental.gross_margin if fundamental else None, settings.treasure_gross_margin_good),
            _score_higher_better(
                fundamental.operating_margin if fundamental else None,
                settings.treasure_operating_margin_good,
            ),
            _score_higher_better(fundamental.free_cash_flow if fundamental else None, max(settings.treasure_free_cash_flow_good, 1.0)),
            _score_lower_better(fundamental.debt_ratio if fundamental else None, settings.treasure_debt_ratio_good),
        ]
    )
    valuation_score = _average(
        [
            _score_lower_better(fundamental.pe_ratio if fundamental else None, settings.treasure_pe_good),
            _score_lower_better(fundamental.pb_ratio if fundamental else None, settings.treasure_pb_good),
            _score_higher_better(
                fundamental.dividend_yield if fundamental else None,
                settings.treasure_dividend_yield_good,
            ),
        ]
    )

    catalyst_strength = min(len(matched_titles) * 20, 80)
    recent_news_count = sum(1 for item in news_items if item.published_at.replace(tzinfo=None) >= recent_cutoff.replace(tzinfo=None))
    catalyst_score = None
    if news_items:
        catalyst_score = round(min(catalyst_strength + recent_news_count * 5, 100), 2)

    stability_score = _average(
        [
            _score_lower_better(fundamental.debt_ratio if fundamental else None, settings.treasure_debt_ratio_good),
            _score_higher_better(fundamental.free_cash_flow if fundamental else None, max(settings.treasure_free_cash_flow_good, 1.0)),
        ]
    )

    component_scores = [growth_score, quality_score, valuation_score, catalyst_score, stability_score]
    available_count = len([score for score in component_scores if score is not None])
    if available_count < settings.treasure_min_required_factors:
        return {
            "treasure_status": "unavailable",
            "treasure_score": None,
            "value_score": None,
            "growth_score": growth_score,
            "quality_score": quality_score,
            "valuation_score": valuation_score,
            "catalyst_score": catalyst_score,
            "value_summary": "基本面與估值資料不足，暫不產生寶藏股結論。",
            "value_risks": {"reasons": ["insufficient_treasure_factors"]},
            "treasure_evidence": {
                "available_factor_count": available_count,
                "required_factor_count": settings.treasure_min_required_factors,
                "matched_news": matched_titles,
            },
        }

    weighted_score = round(
        (growth_score or 0.0) * settings.treasure_weight_growth
        + (quality_score or 0.0) * settings.treasure_weight_quality
        + (valuation_score or 0.0) * settings.treasure_weight_valuation
        + (catalyst_score or 0.0) * settings.treasure_weight_catalyst
        + (stability_score or 0.0) * settings.treasure_weight_stability,
        2,
    )

    risks: list[str] = []
    if fundamental is None:
        risks.append("fundamental_snapshot_unavailable")
    else:
        if fundamental.debt_ratio is not None and fundamental.debt_ratio > settings.treasure_debt_ratio_good:
            risks.append("debt_ratio_above_preferred_range")
        if fundamental.pe_ratio is not None and fundamental.pe_ratio > settings.treasure_pe_good * 1.5:
            risks.append("pe_ratio_elevated")
        if fundamental.pb_ratio is not None and fundamental.pb_ratio > settings.treasure_pb_good * 1.5:
            risks.append("pb_ratio_elevated")
        if fundamental.free_cash_flow is not None and fundamental.free_cash_flow < settings.treasure_free_cash_flow_good:
            risks.append("free_cash_flow_negative")
    if revenue_snapshot is not None:
        if revenue_snapshot.revenue_yoy is not None and revenue_snapshot.revenue_yoy < 0:
            risks.append("revenue_yoy_negative")
        if revenue_snapshot.revenue_mom is not None and revenue_snapshot.revenue_mom < 0:
            risks.append("revenue_mom_negative")
    else:
        risks.append("revenue_snapshot_unavailable")
    if not matched_titles:
        risks.append("catalyst_news_not_detected")

    summary_parts: list[str] = []
    if growth_score is not None:
        summary_parts.append(f"成長分 {growth_score}")
    if quality_score is not None:
        summary_parts.append(f"品質分 {quality_score}")
    if valuation_score is not None:
        summary_parts.append(f"估值分 {valuation_score}")
    if catalyst_score is not None:
        summary_parts.append(f"催化分 {catalyst_score}")
    if matched_titles:
        summary_parts.append(
            "近期新聞關鍵字：" + "、".join(sorted({keyword for item in matched_titles for keyword in item["matched_keywords"]}))
        )
    summary = "；".join(summary_parts) if summary_parts else "寶藏股資料不足。"

    return {
        "treasure_status": "ready",
        "treasure_score": weighted_score,
        "value_score": weighted_score,
        "growth_score": growth_score,
        "quality_score": quality_score,
        "valuation_score": valuation_score,
        "catalyst_score": catalyst_score,
        "value_summary": summary,
        "value_risks": {"reasons": sorted(set(risks))},
        "treasure_evidence": {
            "fundamental": {
                "snapshot_date": fundamental.snapshot_date.isoformat() if fundamental else None,
                "revenue_mom": fundamental.revenue_mom if fundamental else None,
                "eps": fundamental.eps if fundamental else None,
                "roe": fundamental.roe if fundamental else None,
                "gross_margin": fundamental.gross_margin if fundamental else None,
                "operating_margin": fundamental.operating_margin if fundamental else None,
                "free_cash_flow": fundamental.free_cash_flow if fundamental else None,
                "debt_ratio": fundamental.debt_ratio if fundamental else None,
                "pe_ratio": fundamental.pe_ratio if fundamental else None,
                "pb_ratio": fundamental.pb_ratio if fundamental else None,
                "dividend_yield": fundamental.dividend_yield if fundamental else None,
                "source_name": fundamental.source_name if fundamental else None,
                "source_url": fundamental.source_url if fundamental else None,
                "fetched_at": fundamental.fetched_at.isoformat() if fundamental else None,
            },
            "revenue": {
                "revenue_month": revenue_snapshot.revenue_month.isoformat() if revenue_snapshot else None,
                "revenue_yoy": revenue_snapshot.revenue_yoy if revenue_snapshot else None,
                "revenue_mom": revenue_snapshot.revenue_mom if revenue_snapshot else None,
                "monthly_revenue": revenue_snapshot.monthly_revenue if revenue_snapshot else None,
                "source_name": revenue_snapshot.source_name if revenue_snapshot else None,
                "source_url": revenue_snapshot.source_url if revenue_snapshot else None,
                "fetched_at": revenue_snapshot.fetched_at.isoformat() if revenue_snapshot else None,
            },
            "news": {
                "matched_news": matched_titles,
                "recent_news_count": recent_news_count,
            },
            "weights": {
                "growth": settings.treasure_weight_growth,
                "quality": settings.treasure_weight_quality,
                "valuation": settings.treasure_weight_valuation,
                "catalyst": settings.treasure_weight_catalyst,
                "stability": settings.treasure_weight_stability,
            },
        },
    }

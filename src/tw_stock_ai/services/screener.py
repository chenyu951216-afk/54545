from __future__ import annotations

from collections import defaultdict
from datetime import date

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import FundamentalSnapshot, NewsItem, PriceBar, RevenueSnapshot, ScreeningCandidate, ScreeningRun
from tw_stock_ai.services.app_settings import build_effective_settings
from tw_stock_ai.services.ranking_engine import rank_candidates
from tw_stock_ai.services.scoring import build_candidate_payload
from tw_stock_ai.services.short_term_types import CandidateEvaluation
from tw_stock_ai.services.stock_profile import resolve_symbol_name


def run_screening(session: Session) -> ScreeningRun:
    settings = build_effective_settings(session)

    bars = session.scalars(
        select(PriceBar).order_by(PriceBar.symbol.asc(), PriceBar.trade_date.asc())
    ).all()
    grouped: dict[str, list[PriceBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.symbol].append(bar)

    latest_trade_date = max((bar.trade_date for bar in bars), default=date.today())
    run = ScreeningRun(
        as_of_date=latest_trade_date,
        status="completed",
        universe_size=len(grouped),
        notes=None if grouped else "no_price_data_available",
    )
    session.add(run)
    session.flush()

    latest_fundamentals = {
        item.symbol: item
        for item in session.scalars(
            select(FundamentalSnapshot).order_by(
                FundamentalSnapshot.symbol.asc(),
                desc(FundamentalSnapshot.snapshot_date),
            )
        ).all()
    }
    latest_revenues = {
        item.symbol: item
        for item in session.scalars(
            select(RevenueSnapshot).order_by(
                RevenueSnapshot.symbol.asc(),
                desc(RevenueSnapshot.revenue_month),
            )
        ).all()
    }
    grouped_news: dict[str, list[NewsItem]] = defaultdict(list)
    for item in session.scalars(
        select(NewsItem).order_by(NewsItem.published_at.desc())
    ).all():
        if item.symbol:
            grouped_news[item.symbol].append(item)

    evaluated: list[tuple[CandidateEvaluation, dict]] = []
    for symbol, symbol_bars in grouped.items():
        symbol_name = resolve_symbol_name(session, symbol, symbol_bars)
        payload = build_candidate_payload(
            symbol=symbol,
            bars=symbol_bars,
            fundamental=latest_fundamentals.get(symbol),
            min_bars=settings.min_price_bars_for_screening,
            revenue_snapshot=latest_revenues.get(symbol),
            news_items=grouped_news.get(symbol, []),
            symbol_name=symbol_name,
            settings=settings,
        )
        evaluated.append(
            (
                CandidateEvaluation(
                    symbol=symbol,
                    symbol_name=symbol_name,
                    status=payload["status"],
                    as_of_date=run.as_of_date,
                    overall_score=payload["overall_score"],
                    sub_scores=payload["sub_scores"],
                    evidence=payload["evidence"],
                    entry_zone_low=payload["entry_zone_low"],
                    entry_zone_high=payload["entry_zone_high"],
                    stop_loss=payload["stop_loss"],
                    take_profit=payload["take_profit"],
                    take_profit_1=payload["take_profit_1"],
                    take_profit_2=payload["take_profit_2"],
                    risk_reward_ratio=payload["risk_reward_ratio"],
                    holding_days_min=payload["holding_days_min"],
                    holding_days_max=payload["holding_days_max"],
                    risk_flags=payload["risk_flags"],
                ),
                payload,
            )
        )

    ranked = rank_candidates([item for item, _ in evaluated])
    payload_map = {item.symbol: payload for item, payload in evaluated}
    candidates: list[ScreeningCandidate] = []
    for index, candidate in enumerate(ranked, start=1):
        payload = payload_map[candidate.symbol]
        candidates.append(
            ScreeningCandidate(
                run_id=run.id,
                rank_position=index,
                symbol=candidate.symbol,
                symbol_name=candidate.symbol_name,
                status=candidate.status,
                overall_score=candidate.overall_score,
                sub_scores=candidate.sub_scores,
                evidence=candidate.evidence,
                entry_zone_low=candidate.entry_zone_low,
                entry_zone_high=candidate.entry_zone_high,
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profit,
                take_profit_1=candidate.take_profit_1,
                take_profit_2=candidate.take_profit_2,
                risk_reward_ratio=candidate.risk_reward_ratio,
                holding_days_min=candidate.holding_days_min,
                holding_days_max=candidate.holding_days_max,
                risk_flags=candidate.risk_flags,
                treasure_status=payload["treasure_status"],
                treasure_score=payload["treasure_score"],
                value_score=payload["value_score"],
                growth_score=payload["growth_score"],
                quality_score=payload["quality_score"],
                valuation_score=payload["valuation_score"],
                catalyst_score=payload["catalyst_score"],
                value_summary=payload["value_summary"],
                value_risks=payload["value_risks"],
                treasure_evidence=payload["treasure_evidence"],
            )
        )

    session.add_all(candidates)
    session.commit()
    session.refresh(run)
    return run


def get_latest_run(session: Session) -> tuple[ScreeningRun | None, list[ScreeningCandidate]]:
    run = session.scalar(select(ScreeningRun).order_by(ScreeningRun.created_at.desc()))
    if run is None:
        return None, []

    candidates = session.scalars(
        select(ScreeningCandidate)
        .where(ScreeningCandidate.run_id == run.id)
        .order_by(ScreeningCandidate.rank_position.asc(), ScreeningCandidate.symbol.asc())
    ).all()
    return run, candidates

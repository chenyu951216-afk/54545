from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.models import AIAnalysisRecord, Base, Holding, PriceBar, ScreeningCandidate, ScreeningRun
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.prompt_loader import load_prompt_template, render_prompt


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def _make_candidate(run_id: int, rank_position: int, symbol: str) -> ScreeningCandidate:
    return ScreeningCandidate(
        run_id=run_id,
        rank_position=rank_position,
        symbol=symbol,
        symbol_name=f"Name-{symbol}",
        status="ready",
        overall_score=88.0 - rank_position,
        sub_scores={"trend_score": 90.0},
        evidence={"pattern": {"label": "breakout"}, "adx": 25.0},
        entry_zone_low=100.0,
        entry_zone_high=102.0,
        stop_loss=97.0,
        take_profit=108.0,
        take_profit_1=108.0,
        take_profit_2=112.0,
        risk_reward_ratio=1.8,
        holding_days_min=3,
        holding_days_max=10,
        risk_flags={"reasons": []},
        treasure_status="ready",
        treasure_score=70.0,
        value_score=70.0,
        growth_score=75.0,
        quality_score=72.0,
        valuation_score=68.0,
        catalyst_score=65.0,
        value_summary="value summary",
        value_risks={"reasons": []},
        treasure_evidence={
            "fundamental": {"eps": 8.0, "roe": 20.0, "gross_margin": 50.0, "operating_margin": 35.0},
            "news": {"matched_news": [{"title": "AI order update", "matched_keywords": ["AI"]}]},
        },
    )


def test_prompt_templates_are_loaded_from_files() -> None:
    template = load_prompt_template("candidate_news_summary")
    assert "Only summarize the provided evidence." in template

    prompt = render_prompt(
        "candidate_selection_reason",
        {"symbol": "2330", "symbol_name": "TSMC", "evidence_json": {"score": 90}},
    )
    assert "2330" in prompt
    assert '"score": 90' in prompt


def test_ai_analysis_only_runs_for_top_candidates() -> None:
    with make_session() as session:
        run = ScreeningRun(
            as_of_date=date(2026, 4, 18),
            status="completed",
            universe_size=2,
            notes=None,
        )
        session.add(run)
        session.flush()
        session.add_all([_make_candidate(run.id, 1, "2330"), _make_candidate(run.id, 2, "2317")])
        session.commit()

        service = AIAnalysisService()
        analyses = service.analyze_top_candidates(session, run.id)

        assert len(analyses) == 8
        records = session.scalars(select(AIAnalysisRecord)).all()
        assert len(records) == 8
        assert all(record.target_type == "screening_candidate" for record in records)


def test_holding_ai_analysis_returns_evidence_insufficient_when_data_is_missing() -> None:
    with make_session() as session:
        holding = Holding(symbol="2330", quantity=1000, average_cost=100.0, note=None)
        session.add(holding)
        session.commit()
        session.refresh(holding)

        service = AIAnalysisService()
        analyses = service.analyze_holding(session, holding.id)

        assert len(analyses) == 1
        assert analyses[0].analysis_kind == "holding_trend_review"
        assert "evidence insufficient" in analyses[0].summary or analyses[0].summary.startswith("持股趨勢檢查")

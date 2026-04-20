from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.models import AIAnalysisRecord, Base, ScreeningCandidate, ScreeningRun, UsageEvent
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.app_settings import save_settings
from tw_stock_ai.services.cost_control import CostControlService
from tw_stock_ai.services.rate_limits import RateLimitExceededError, RateLimitService
from tw_stock_ai.services.usage_tracking import UsageTracker


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def test_rate_limit_service_blocks_after_limit() -> None:
    with make_session() as session:
        save_settings(
            session,
            {
                "api_rate_limit_window_minutes": "60",
                "rate_limit_screening_runs_per_window": "2",
            },
        )
        rate_limits = RateLimitService()
        tracker = UsageTracker()
        now = datetime.now(timezone.utc)
        tracker.record(session, event_type="api_call", operation="manual_screening_run", provider="internal", status="completed", occurred_at=now)
        tracker.record(session, event_type="api_call", operation="manual_screening_run", provider="internal", status="completed", occurred_at=now)

        try:
            rate_limits.enforce(session, operation="manual_screening_run", limit=2)
            assert False, "expected rate limit to raise"
        except RateLimitExceededError:
            assert True


def test_ai_analysis_uses_cache_for_same_evidence() -> None:
    with make_session() as session:
        run = ScreeningRun(as_of_date=date(2026, 4, 19), status="completed", universe_size=1, notes=None)
        session.add(run)
        session.flush()
        candidate = ScreeningCandidate(
            run_id=run.id,
            rank_position=1,
            symbol="2330",
            symbol_name="TSMC",
            status="ready",
            overall_score=88.0,
            sub_scores={"trend_score": 90.0, "momentum_score": 80.0},
            evidence={"pattern": {"label": "breakout"}, "latest_close": 950.0},
            entry_zone_low=945.0,
            entry_zone_high=955.0,
            stop_loss=920.0,
            take_profit=1000.0,
            take_profit_1=1000.0,
            take_profit_2=1030.0,
            risk_reward_ratio=1.8,
            holding_days_min=3,
            holding_days_max=10,
            risk_flags={"reasons": []},
            treasure_status="ready",
            treasure_score=70.0,
            value_score=70.0,
            growth_score=72.0,
            quality_score=75.0,
            valuation_score=68.0,
            catalyst_score=66.0,
            value_summary="value summary",
            value_risks={"reasons": []},
            treasure_evidence={"fundamental": {"eps": 7.0}, "news": {"matched_news": []}},
        )
        session.add(candidate)
        session.commit()

        service = AIAnalysisService()
        first = service.analyze_top_candidates(session, run.id)
        first_count = session.scalar(select(func.count(AIAnalysisRecord.id)).where(AIAnalysisRecord.target_id == candidate.id))
        second = service.analyze_top_candidates(session, run.id)
        second_count = session.scalar(select(func.count(AIAnalysisRecord.id)).where(AIAnalysisRecord.target_id == candidate.id))

        assert len(first) >= 1
        assert len(second) >= 1
        assert first_count == second_count


def test_cost_snapshot_aggregates_usage_and_ai_cost() -> None:
    with make_session() as session:
        now = datetime.now(timezone.utc)
        session.add(
            AIAnalysisRecord(
                target_type="screening_candidate",
                target_id=1,
                symbol="2330",
                analysis_kind="candidate_selection_reason",
                prompt_name="candidate_selection_reason",
                provider="fallback",
                model_name="fallback-v1",
                status="completed",
                summary="ok",
                details={},
                evidence_snapshot={},
                input_tokens=10,
                output_tokens=10,
                estimated_cost_twd=12.5,
                fallback_used=True,
                generated_at=now,
            )
        )
        UsageTracker().record(
            session,
            event_type="notification_send",
            operation="discord_report_send",
            provider="discord",
            status="sent",
            estimated_cost_twd=1.2,
            occurred_at=now,
        )
        session.commit()

        snapshot = CostControlService().build_snapshot(session)

        assert snapshot["ai_actual_cost_twd"] == 12.5
        assert snapshot["monthly_estimated_cost_twd"] >= 13.7
        assert snapshot["monthly_usage"]["notifications"] >= 1

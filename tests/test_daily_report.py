from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from urllib import error

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import Base, DailyReportRun, DiscordDeliveryLog, PriceBar
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.daily_report import DailyReportGenerator
from tw_stock_ai.services.jobs import run_daily_screening_and_push


def make_session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, future=True)
    return local_session()


def _make_bar(index: int, *, volume: int = 800000, volume_step: int = 2000) -> PriceBar:
    trade_date = date(2026, 1, 1) + timedelta(days=index)
    close = 100 + (index * 0.9)
    return PriceBar(
        symbol="2330",
        trade_date=trade_date,
        open=close - 0.8,
        high=close + 1.4,
        low=close - 1.0,
        close=close,
        volume=volume + (index * volume_step),
        source_name="test",
        source_url="https://example.com",
        fetched_at=datetime.now(timezone.utc),
        raw_payload={"symbol_name": "TSMC"},
    )


def test_daily_report_generator_outputs_no_qualified_picks_message() -> None:
    with make_session() as session:
        report = DailyReportRun(
            report_kind="discord_top_picks",
            report_date=date(2026, 4, 20),
            trigger_source="test",
            status="running",
            qualified_count=0,
            top_n=5,
            rendered_content="",
            payload_json={},
        )
        session.add(report)
        session.flush()

        generator = DailyReportGenerator(top_n=5, reason_max_length=120, risk_max_length=120)
        generator.populate_report_run(
            session,
            report_run=report,
            screening_run_id=None,
            report_date=date(2026, 4, 20),
        )

        assert report.qualified_count == 0
        assert "today no qualified picks" in report.rendered_content
        assert report.payload_json["today_no_qualified_picks"] is True


def test_daily_screening_and_push_retries_then_logs_success(monkeypatch) -> None:
    with make_session() as session:
        session.add_all([_make_bar(index) for index in range(140)])
        last_bar = session.scalar(
            select(PriceBar).where(PriceBar.trade_date == date(2026, 5, 20))
        )
        if last_bar is not None:
            last_bar.close = 232.0
            last_bar.high = 233.0
            last_bar.open = 228.0
            last_bar.volume = 2000000
        session.commit()

        settings = get_settings()
        monkeypatch.setattr(settings, "discord_enabled", True)
        monkeypatch.setattr(settings, "discord_webhook_url", "https://discord.example/webhook/abc123456789")
        monkeypatch.setattr(settings, "discord_retry_attempts", 2)
        monkeypatch.setattr(settings, "discord_retry_backoff_seconds", 0.0)
        monkeypatch.setattr(settings, "discord_daily_report_top_n", 5)
        monkeypatch.setattr(settings, "ai_top_n_candidates", 5)

        monkeypatch.setattr(AIAnalysisService, "analyze_top_candidates", lambda self, db, run_id: [])

        calls = {"count": 0}

        def fake_post_payload(self, payload: dict) -> tuple[int, str]:
            calls["count"] += 1
            if calls["count"] == 1:
                raise error.URLError("temporary_network")
            return 204, ""

        monkeypatch.setattr("tw_stock_ai.services.discord.DiscordWebhookSender._post_payload", fake_post_payload)

        report = run_daily_screening_and_push(session, trigger_source="test")
        logs = session.scalars(
            select(DiscordDeliveryLog)
            .where(DiscordDeliveryLog.report_run_id == report.id)
            .order_by(DiscordDeliveryLog.attempt_no.asc())
        ).all()

        assert report.status == "sent"
        assert report.qualified_count >= 1
        assert calls["count"] == 2
        assert [item.status for item in logs] == ["failed", "sent"]
        assert "2330" in report.rendered_content

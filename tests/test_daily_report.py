from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from urllib import error

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import Base, DailyReportRun, DiscordDeliveryLog, Holding, PriceBar, ScreeningRun
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.daily_report import DailyReportGenerator
from tw_stock_ai.services.discord import DiscordWebhookSender
from tw_stock_ai.services.jobs import (
    _normalize_daily_report_trigger_source,
    refresh_intraday_news,
    run_daily_screening_and_push,
)


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


def test_discord_sender_stops_retrying_on_fatal_403(monkeypatch) -> None:
    with make_session() as session:
        report = DailyReportRun(
            report_kind="discord_top_picks",
            report_date=date(2026, 4, 20),
            trigger_source="test",
            status="prepared",
            qualified_count=1,
            top_n=5,
            rendered_content="test content",
            payload_json={},
        )
        session.add(report)
        session.commit()
        session.refresh(report)

        settings = get_settings()
        monkeypatch.setattr(settings, "discord_enabled", True)
        monkeypatch.setattr(settings, "discord_webhook_url", "https://discord.example/webhook/abc123456789")
        monkeypatch.setattr(settings, "discord_retry_attempts", 3)
        monkeypatch.setattr(settings, "discord_retry_backoff_seconds", 0.0)

        attempts = {"count": 0}

        def fatal_403(self, payload: dict) -> tuple[int, str]:
            attempts["count"] += 1
            raise error.HTTPError(
                url="https://discord.example/webhook/abc123456789",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=BytesIO(b"missing access"),
            )

        monkeypatch.setattr("tw_stock_ai.services.discord.DiscordWebhookSender._post_payload", fatal_403)

        result = DiscordWebhookSender(settings=settings).send_report(session, report)

        assert result.status == "failed"
        assert result.attempts == 1
        assert attempts["count"] == 1
        assert report.error_detail is not None
        assert "http_error:403" in report.error_detail


def test_daily_report_trigger_source_is_compacted_to_fit_column() -> None:
    compacted = _normalize_daily_report_trigger_source("ui_manual_dispatch:fallback_prepare")

    assert compacted == "ui_dispatch:fb_prepare"
    assert len(compacted) <= 30


def test_daily_screening_and_push_stores_short_trigger_source(monkeypatch) -> None:
    with make_session() as session:
        session.add_all([_make_bar(index) for index in range(140)])
        session.commit()

        settings = get_settings()
        monkeypatch.setattr(settings, "discord_enabled", False)
        monkeypatch.setattr(settings, "discord_daily_report_top_n", 5)
        monkeypatch.setattr(settings, "ai_top_n_candidates", 5)

        monkeypatch.setattr(AIAnalysisService, "analyze_top_candidates", lambda self, db, run_id: [])

        report = run_daily_screening_and_push(session, trigger_source="ui_manual_dispatch")

        assert report.trigger_source == "ui_dispatch:prep"
        assert len(report.trigger_source) <= 30


def test_refresh_intraday_news_targets_latest_candidates_and_holdings(monkeypatch) -> None:
    with make_session() as session:
        session.add(
            ScreeningRun(
                as_of_date=date(2026, 4, 21),
                status="completed",
                universe_size=10,
                notes=None,
            )
        )
        session.flush()
        latest_run = session.scalar(select(ScreeningRun).order_by(ScreeningRun.id.desc()))
        assert latest_run is not None
        session.add(Holding(symbol="2317", quantity=1, average_cost=100.0))
        session.commit()

        captured: dict = {}

        def fake_collect(session_obj, run_id: int, top_n: int) -> list[str]:  # noqa: ANN001
            captured["run_id"] = run_id
            captured["top_n"] = top_n
            return ["2330", "2454"]

        class FakeRefreshRun:
            status = "completed"

        def fake_refresh_all(self, session_obj, requests=None, *, trigger_source="manual", dataset_names=None):  # noqa: ANN001
            captured["trigger_source"] = trigger_source
            captured["dataset_names"] = dataset_names
            captured["symbols"] = requests["news"].symbols
            captured["force_refresh"] = requests["news"].force_refresh
            return FakeRefreshRun()

        monkeypatch.setattr("tw_stock_ai.services.jobs._collect_deep_refresh_symbols", fake_collect)
        monkeypatch.setattr("tw_stock_ai.services.jobs.DataRefreshCoordinator.refresh_all", fake_refresh_all)

        result = refresh_intraday_news(session, trigger_source="sched_news")

        assert result.status == "completed"
        assert captured["run_id"] == latest_run.id
        assert captured["top_n"] == get_settings().news_fetch_max_symbols_per_run
        assert captured["trigger_source"] == "sched_news"
        assert captured["dataset_names"] == ["news"]
        assert captured["symbols"] == ["2317", "2330", "2454"]
        assert captured["force_refresh"] is True

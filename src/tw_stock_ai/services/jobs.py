from __future__ import annotations

from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.config import get_settings
from tw_stock_ai.db import SessionLocal
from tw_stock_ai.models import DailyReportRun, DiscordDeliveryLog, Holding, ScreeningCandidate, ScreeningRun
from tw_stock_ai.notifiers.registry import build_default_notifier
from tw_stock_ai.schemas import DailyReportRunRead
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.app_settings import build_effective_settings
from tw_stock_ai.services.cost_control import CostControlService
from tw_stock_ai.services.data_refresh import DataRefreshCoordinator
from tw_stock_ai.services.daily_report import DailyReportGenerator, serialize_report_run
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.logging_config import get_logger
from tw_stock_ai.services.portfolio import enrich_holding
from tw_stock_ai.services.position_monitor import PositionMonitorService
from tw_stock_ai.services.rate_limits import RateLimitExceededError, RateLimitService
from tw_stock_ai.services.screener import run_screening

logger = get_logger("tw_stock_ai.jobs")


def _normalize_daily_report_trigger_source(trigger_source: str) -> str:
    normalized = (trigger_source or "").strip() or "manual"
    segment_replacements = {
        "worker_startup_bootstrap": "worker_boot",
        "scheduler_prewarm": "sched_prewarm",
        "scheduler_push": "sched_push",
        "manual_api": "api_manual",
        "ui_manual_dispatch": "ui_dispatch",
        "ui_manual": "ui_manual",
        "fallback_prepare": "fb_prepare",
        "prepare": "prep",
        "dispatch": "push",
    }

    normalized = ":".join(
        segment_replacements.get(part, part)
        for part in normalized.replace("-", "_").split(":")
    )

    if len(normalized) <= 30:
        return normalized

    compact_parts: list[str] = []
    for part in normalized.replace("-", "_").split(":"):
        tokens = [token for token in part.split("_") if token]
        if not tokens:
            continue
        compact_parts.append("_".join(token[:6] for token in tokens))

    compacted = ":".join(compact_parts) or "manual"
    if len(compacted) <= 30:
        return compacted

    return compacted[:30]


def build_scheduler() -> BackgroundScheduler:
    with SessionLocal() as session:
        settings = build_effective_settings(session)
    scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)
    logger.info(
        "scheduler_configured timezone=%s prewarm=%02d:%02d push=%02d:%02d weekdays=%s",
        settings.scheduler_timezone,
        settings.prewarm_hour,
        settings.prewarm_minute,
        settings.screening_hour,
        settings.screening_minute,
        settings.screening_weekdays,
    )
    scheduler.add_job(
        func=run_daily_prewarm_job,
        trigger="cron",
        day_of_week=settings.screening_weekdays,
        hour=settings.prewarm_hour,
        minute=settings.prewarm_minute,
        id="daily-market-prewarm",
        replace_existing=True,
    )
    scheduler.add_job(
        func=run_daily_report_push_job,
        trigger="cron",
        day_of_week=settings.screening_weekdays,
        hour=settings.screening_hour,
        minute=settings.screening_minute,
        id="daily-screening-discord-push",
        replace_existing=True,
    )
    return scheduler


def run_daily_prewarm_job() -> dict:
    with SessionLocal() as session:
        report = prepare_daily_screening_and_analysis(session, trigger_source="scheduler_prewarm")
        return DailyReportRunRead.model_validate(
            serialize_report_run(report, _load_delivery_logs(session, report.id))
        ).model_dump()


def run_daily_report_push_job() -> dict:
    with SessionLocal() as session:
        report = dispatch_prepared_daily_report(session, trigger_source="scheduler_push")
        return DailyReportRunRead.model_validate(
            serialize_report_run(report, _load_delivery_logs(session, report.id))
        ).model_dump()


def maybe_run_startup_bootstrap() -> DailyReportRun | None:
    with SessionLocal() as session:
        settings = build_effective_settings(session)
        if not settings.startup_bootstrap_enabled:
            logger.info("startup_bootstrap_skipped reason=disabled")
            return None

        latest_report = get_latest_daily_report(session, report_date=date.today())
        latest_screening = session.scalar(
            select(ScreeningRun)
            .where(ScreeningRun.as_of_date == date.today())
            .order_by(desc(ScreeningRun.created_at), desc(ScreeningRun.id))
        )
        if latest_report is not None and latest_screening is not None:
            logger.info(
                "startup_bootstrap_skipped reason=today_already_prepared report_id=%s screening_id=%s",
                latest_report.id,
                latest_screening.id,
            )
            return None

        logger.info(
            "startup_bootstrap_running force_refresh=%s",
            settings.startup_bootstrap_force_refresh,
        )
        return prepare_daily_screening_and_analysis(
            session,
            trigger_source="worker_startup_bootstrap",
            force_refresh=settings.startup_bootstrap_force_refresh,
        )


def refresh_market_data(
    session: Session,
    *,
    trigger_source: str,
    force_refresh: bool = False,
) -> None:
    coordinator = DataRefreshCoordinator()
    refresh_run = coordinator.refresh_default(
        session,
        trigger_source=trigger_source,
        force_refresh=force_refresh,
    )
    logger.info(
        "market_data_refreshed trigger_source=%s run_id=%s status=%s",
        trigger_source,
        refresh_run.id,
        refresh_run.status,
    )


def refresh_symbols_data(
    session: Session,
    *,
    symbols: list[str],
    trigger_source: str,
    force_refresh: bool = False,
) -> None:
    normalized_symbols = sorted({symbol.strip() for symbol in symbols if symbol and symbol.strip()})
    if not normalized_symbols:
        return

    coordinator = DataRefreshCoordinator()
    requests = coordinator.build_default_requests(session, force_refresh=force_refresh)
    for dataset in ("price", "volume", "news", "revenue", "fundamentals"):
        requests[dataset].symbols = normalized_symbols
    refresh_run = coordinator.refresh_all(
        session,
        requests=requests,
        trigger_source=trigger_source,
    )
    logger.info(
        "symbol_scoped_market_data_refreshed trigger_source=%s run_id=%s status=%s symbols=%s",
        trigger_source,
        refresh_run.id,
        refresh_run.status,
        normalized_symbols,
    )


def prepare_daily_screening_and_analysis(
    session: Session,
    *,
    trigger_source: str,
    force_refresh: bool = False,
) -> DailyReportRun:
    report_trigger_source = _normalize_daily_report_trigger_source(trigger_source)
    settings = build_effective_settings(session)
    flags = FeatureFlagService(settings)
    generator = DailyReportGenerator(
        top_n=settings.discord_daily_report_top_n,
        reason_max_length=settings.discord_reason_max_length,
        risk_max_length=settings.discord_risk_max_length,
    )
    report = DailyReportRun(
        report_kind="discord_top_picks",
        report_date=date.today(),
        trigger_source=report_trigger_source,
        status="running",
        qualified_count=0,
        top_n=settings.discord_daily_report_top_n,
        rendered_content="",
        payload_json={},
    )
    session.add(report)
    session.flush()

    try:
        if not flags.is_enabled("daily_report", session):
            report.status = "skipped"
            report.error_detail = "feature_daily_report_enabled=false"
            report.rendered_content = "daily report disabled"
            report.payload_json = {"reason": "feature_daily_report_enabled=false"}
            session.commit()
            session.refresh(report)
            return report

        refresh_market_data(
            session,
            trigger_source=f"{trigger_source}:data_refresh",
            force_refresh=force_refresh,
        )
        screening_run = run_screening(session)
        logger.info(
            "screening_completed run_id=%s trigger_source=%s stage=initial",
            screening_run.id,
            trigger_source,
        )

        deep_refresh_symbols = _collect_deep_refresh_symbols(session, screening_run.id, settings.discord_daily_report_top_n)
        if deep_refresh_symbols:
            refresh_symbols_data(
                session,
                symbols=deep_refresh_symbols,
                trigger_source=f"{trigger_source}:deep_refresh",
                force_refresh=force_refresh,
            )
            screening_run = run_screening(session)
            logger.info(
                "screening_completed run_id=%s trigger_source=%s stage=deep_refresh_rerun symbols=%s",
                screening_run.id,
                trigger_source,
                deep_refresh_symbols,
            )

        ai_service = AIAnalysisService(settings=settings)
        ai_service.analyze_top_candidates(session, screening_run.id)

        if _holding_count(session) > 0:
            PositionMonitorService().monitor_all_positions(session)
            if flags.is_enabled("holding_ai_analysis", session):
                for holding in session.scalars(select(Holding).order_by(Holding.created_at.desc())).all():
                    ai_service.analyze_holding(session, holding.id)

        generator.populate_report_run(
            session,
            report_run=report,
            screening_run_id=screening_run.id,
            report_date=screening_run.as_of_date,
        )
        logger.info(
            "daily_report_prepared report_run_id=%s screening_run_id=%s trigger_source=%s raw_trigger_source=%s",
            report.id,
            screening_run.id,
            report_trigger_source,
            trigger_source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "daily_screening_prepare_failed trigger_source=%s raw_trigger_source=%s error=%s",
            report_trigger_source,
            trigger_source,
            exc,
        )
        report.status = "failed"
        report.error_detail = str(exc)
        report.rendered_content = (
            "鍙拌偂鐭窔姣忔棩鎺ㄦ挱\n"
            f"鏃ユ湡: {report.report_date.isoformat()}\n"
            "daily report preparation failed"
        )
        report.payload_json = {
            "report_date": report.report_date.isoformat(),
            "screening_run_id": report.screening_run_id,
            "qualified_count": 0,
            "top_n": settings.discord_daily_report_top_n,
            "items": [],
            "job_error": str(exc),
        }
    session.commit()
    session.refresh(report)
    return report


def dispatch_prepared_daily_report(
    session: Session,
    *,
    trigger_source: str,
    report_run: DailyReportRun | None = None,
) -> DailyReportRun:
    settings = build_effective_settings(session)
    flags = FeatureFlagService(settings)
    rate_limits = RateLimitService(settings)
    notifier = build_default_notifier(settings)
    cost_control = CostControlService(settings)

    report = report_run or get_latest_daily_report(session, report_date=date.today())
    if report is None or report.status == "failed":
        report = prepare_daily_screening_and_analysis(session, trigger_source=f"{trigger_source}:fallback_prepare")

    try:
        if flags.is_enabled("cost_guardrails", session):
            rate_limits.enforce(
                session,
                operation="discord_report_run",
                limit=settings.rate_limit_discord_reports_per_window,
            )

        if not flags.is_enabled("daily_report", session):
            report.status = "skipped"
            report.error_detail = "feature_daily_report_enabled=false"
        elif flags.is_enabled("discord_notifications", session) and cost_control.within_overall_budget(
            session,
            settings.estimated_notification_cost_per_send_twd,
        ):
            notifier.send(session, report_run=report)
            logger.info("notification_sent report_run_id=%s status=%s", report.id, report.status)
        else:
            report.status = "skipped"
            report.error_detail = "discord_notifications_disabled_or_budget_blocked"

        rate_limits.record(
            session,
            operation="discord_report_run",
            status=report.status,
            metadata={"trigger_source": trigger_source, "qualified_count": report.qualified_count},
        )
    except RateLimitExceededError as exc:
        report.status = "skipped"
        report.error_detail = str(exc)
        rate_limits.record(
            session,
            operation="discord_report_run",
            status="rate_limited",
            metadata={"trigger_source": trigger_source, "error": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("daily_report_dispatch_failed trigger_source=%s error=%s", trigger_source, exc)
        report.status = "failed"
        report.error_detail = str(exc)
        rate_limits.record(
            session,
            operation="discord_report_run",
            status="failed",
            metadata={"trigger_source": trigger_source, "error": str(exc)},
        )

    session.commit()
    session.refresh(report)
    return report


def run_daily_screening_and_push(session: Session, *, trigger_source: str) -> DailyReportRun:
    prepared = prepare_daily_screening_and_analysis(session, trigger_source=f"{trigger_source}:prepare")
    return dispatch_prepared_daily_report(
        session,
        trigger_source=f"{trigger_source}:dispatch",
        report_run=prepared,
    )


def get_latest_daily_report(session: Session, *, report_date: date | None = None) -> DailyReportRun | None:
    stmt = select(DailyReportRun)
    if report_date is not None:
        stmt = stmt.where(DailyReportRun.report_date == report_date)
    return session.scalar(stmt.order_by(desc(DailyReportRun.created_at), desc(DailyReportRun.id)))


def _load_delivery_logs(session: Session, report_run_id: int) -> list[DiscordDeliveryLog]:
    return session.scalars(
        select(DiscordDeliveryLog)
        .where(DiscordDeliveryLog.report_run_id == report_run_id)
        .order_by(DiscordDeliveryLog.attempt_no.asc(), DiscordDeliveryLog.created_at.asc())
    ).all()


def serialize_daily_report(session: Session, report: DailyReportRun) -> DailyReportRunRead:
    return DailyReportRunRead.model_validate(
        serialize_report_run(report, _load_delivery_logs(session, report.id))
    )


def _collect_deep_refresh_symbols(session: Session, run_id: int, top_n: int) -> list[str]:
    candidate_symbols = session.scalars(
        select(ScreeningCandidate.symbol)
        .where(
            ScreeningCandidate.run_id == run_id,
            ScreeningCandidate.status == "ready",
        )
        .order_by(ScreeningCandidate.rank_position.asc(), ScreeningCandidate.symbol.asc())
        .limit(top_n)
    ).all()
    holding_symbols = session.scalars(select(Holding.symbol).order_by(Holding.created_at.desc())).all()
    merged = sorted({symbol.strip() for symbol in [*candidate_symbols, *holding_symbols] if symbol and symbol.strip()})
    return merged


def _holding_count(session: Session) -> int:
    return len(session.scalars(select(Holding.id)).all())

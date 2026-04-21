from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from tw_stock_ai.models import DailyReportRun, DataRefreshItem, DataRefreshRun, Holding, PriceBar, ScreeningRun
from tw_stock_ai.services.app_settings import build_effective_settings


class StartupCheckService:
    def build_snapshot(self, session: Session) -> dict:
        settings = build_effective_settings(session)
        now = datetime.now(timezone.utc)
        stale_cutoff = now.date() - timedelta(days=settings.startup_check_recent_days)

        checks: list[dict] = []

        try:
            session.execute(text("SELECT 1"))
            checks.append({"name": "database", "status": "ready", "detail": "database_connected"})
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": "database", "status": "failed", "detail": str(exc)})

        latest_refresh = session.scalar(
            select(DataRefreshRun).order_by(desc(DataRefreshRun.created_at), desc(DataRefreshRun.id))
        )
        latest_screening = session.scalar(
            select(ScreeningRun).order_by(desc(ScreeningRun.created_at), desc(ScreeningRun.id))
        )
        latest_report = session.scalar(
            select(DailyReportRun).order_by(desc(DailyReportRun.created_at), desc(DailyReportRun.id))
        )
        latest_price_date = session.scalar(select(func.max(PriceBar.trade_date)))

        refresh_items: list[DataRefreshItem] = []
        if latest_refresh is not None:
            refresh_items = session.scalars(
                select(DataRefreshItem)
                .where(DataRefreshItem.run_id == latest_refresh.id)
                .order_by(DataRefreshItem.dataset.asc(), DataRefreshItem.id.asc())
            ).all()

        checks.append(
            {
                "name": "scheduler_mode",
                "status": "ready",
                "detail": "worker_mode" if settings.enable_scheduler else "web_mode",
            }
        )
        checks.append(
            {
                "name": "market_data_refresh",
                "status": (
                    "ready"
                    if latest_refresh is not None and latest_refresh.status == "completed"
                    else "warning"
                    if latest_refresh is not None
                    else "warning"
                ),
                "detail": latest_refresh.status if latest_refresh is not None else "no_refresh_run",
                "metadata": {
                    "run_id": latest_refresh.id if latest_refresh is not None else None,
                    "started_at": latest_refresh.started_at.isoformat() if latest_refresh is not None else None,
                    "datasets": {
                        item.dataset: {
                            "status": item.status,
                            "records_stored": item.records_stored,
                            "detail": item.detail,
                        }
                        for item in refresh_items
                    },
                },
            }
        )
        checks.append(
            {
                "name": "latest_price_bars",
                "status": (
                    "ready"
                    if latest_price_date is not None and latest_price_date >= stale_cutoff
                    else "warning"
                ),
                "detail": latest_price_date.isoformat() if latest_price_date is not None else "no_price_bars",
            }
        )
        checks.append(
            {
                "name": "latest_screening",
                "status": (
                    "ready"
                    if latest_screening is not None and latest_screening.as_of_date >= stale_cutoff
                    else "warning"
                ),
                "detail": latest_screening.status if latest_screening is not None else "no_screening_run",
                "metadata": {
                    "run_id": latest_screening.id if latest_screening is not None else None,
                    "as_of_date": latest_screening.as_of_date.isoformat() if latest_screening is not None else None,
                    "universe_size": latest_screening.universe_size if latest_screening is not None else 0,
                },
            }
        )
        checks.append(
            {
                "name": "latest_daily_report",
                "status": (
                    "ready"
                    if latest_report is not None and latest_report.report_date >= stale_cutoff
                    else "warning"
                ),
                "detail": latest_report.status if latest_report is not None else "no_daily_report",
                "metadata": {
                    "report_id": latest_report.id if latest_report is not None else None,
                    "report_date": latest_report.report_date.isoformat() if latest_report is not None else None,
                    "qualified_count": latest_report.qualified_count if latest_report is not None else 0,
                },
            }
        )
        checks.append(
            {
                "name": "openai",
                "status": "ready" if settings.ai_enabled and bool(settings.openai_api_key) else "warning",
                "detail": (
                    "configured"
                    if settings.ai_enabled and bool(settings.openai_api_key)
                    else "ai_disabled_or_key_missing"
                ),
            }
        )
        checks.append(
            {
                "name": "fugle",
                "status": "ready" if bool(settings.fugle_api_key) else "warning",
                "detail": "configured" if bool(settings.fugle_api_key) else "fugle_api_key_missing",
            }
        )
        checks.append(
            {
                "name": "discord",
                "status": "ready" if settings.discord_enabled and bool(settings.discord_webhook_url) else "warning",
                "detail": (
                    "configured"
                    if settings.discord_enabled and bool(settings.discord_webhook_url)
                    else "discord_disabled_or_webhook_missing"
                ),
            }
        )

        holding_symbols = session.scalars(select(Holding.symbol).order_by(Holding.created_at.desc())).all()
        holding_coverage: list[dict] = []
        for symbol in holding_symbols:
            latest_holding_price = session.scalar(
                select(func.max(PriceBar.trade_date)).where(PriceBar.symbol == symbol)
            )
            holding_coverage.append(
                {
                    "symbol": symbol,
                    "latest_price_date": latest_holding_price.isoformat() if latest_holding_price is not None else None,
                    "status": (
                        "ready"
                        if latest_holding_price is not None and latest_holding_price >= stale_cutoff
                        else "warning"
                    ),
                }
            )

        status_priority = {"failed": 3, "warning": 2, "ready": 1}
        overall_status = "ready"
        for check in checks:
            if status_priority.get(check["status"], 0) > status_priority[overall_status]:
                overall_status = check["status"]

        return {
            "generated_at": now.isoformat(),
            "overall_status": overall_status,
            "checks": checks,
            "holding_coverage": holding_coverage,
            "providers": {
                "price": settings.price_data_provider,
                "volume": settings.volume_data_provider,
                "news": settings.news_data_provider,
                "revenue": settings.revenue_data_provider,
                "fundamentals": settings.fundamentals_data_provider,
                "market_calendar": settings.market_calendar_provider,
            },
            "schedule": {
                "timezone": settings.scheduler_timezone,
                "startup_bootstrap_enabled": settings.startup_bootstrap_enabled,
                "prewarm_time": f"{int(settings.prewarm_hour):02d}:{int(settings.prewarm_minute):02d}",
                "close_refresh_enabled": settings.close_refresh_enabled,
                "close_refresh_time": f"{int(settings.close_refresh_hour):02d}:{int(settings.close_refresh_minute):02d}",
                "push_time": f"{int(settings.screening_hour):02d}:{int(settings.screening_minute):02d}",
                "news_polling_enabled": settings.news_polling_enabled,
                "news_poll_interval_minutes": settings.news_poll_interval_minutes,
                "weekdays": settings.screening_weekdays,
            },
        }

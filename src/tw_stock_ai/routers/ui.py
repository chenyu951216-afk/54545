from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select

from tw_stock_ai.db import SessionLocal
from tw_stock_ai.models import (
    DailyReportRun,
    DataRefreshRun,
    FundamentalSnapshot,
    Holding,
    NewsItem,
    PriceBar,
    RevenueSnapshot,
    ScreeningCandidate,
    ScreeningRun,
)
from tw_stock_ai.services.ai_analysis import AIAnalysisService
from tw_stock_ai.services.app_settings import build_effective_settings, get_settings_for_ui, save_settings
from tw_stock_ai.services.cost_control import CostControlService
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.jobs import (
    dispatch_prepared_daily_report,
    get_latest_daily_report,
    refresh_market_data,
    refresh_symbols_data,
    serialize_daily_report,
)
from tw_stock_ai.services.logging_config import get_logger
from tw_stock_ai.services.news_window import premium_news_window_active
from tw_stock_ai.services.portfolio import enrich_holding
from tw_stock_ai.services.position_monitor import PositionMonitorService
from tw_stock_ai.services.rate_limits import RateLimitExceededError, RateLimitService
from tw_stock_ai.services.screener import get_latest_run, run_screening
from tw_stock_ai.services.startup_check import StartupCheckService

PACKAGE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
logger = get_logger("tw_stock_ai.ui")

router = APIRouter()


def _local_timezone(settings) -> ZoneInfo:
    return ZoneInfo(settings.scheduler_timezone)


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_datetime_local(value: datetime | str | None, settings, *, fallback: str = "unavailable") -> str:
    parsed = _coerce_datetime(value)
    if parsed is None:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    localized = parsed.astimezone(_local_timezone(settings))
    return f"{localized.strftime('%Y-%m-%d %H:%M:%S')} {settings.scheduler_timezone}"


def _build_run_view(run: ScreeningRun | None, settings) -> dict | None:
    if run is None:
        return None
    return {
        "trade_date_display": run.as_of_date.isoformat(),
        "status": run.status,
        "universe_size": run.universe_size,
        "updated_at_display": _format_datetime_local(run.created_at, settings),
    }


def _build_refresh_view(refresh_run: DataRefreshRun | None, settings) -> dict | None:
    if refresh_run is None:
        return None
    return {
        "status": refresh_run.status,
        "started_at_display": _format_datetime_local(refresh_run.started_at, settings),
        "completed_at_display": _format_datetime_local(refresh_run.completed_at, settings),
        "trigger_source": refresh_run.trigger_source,
    }


def _build_report_view(session, report: DailyReportRun | None, settings) -> dict | None:
    if report is None:
        return None
    view = serialize_daily_report(session, report).model_dump()
    view["report_date_display"] = report.report_date.isoformat()
    view["created_at_display"] = _format_datetime_local(view.get("created_at"), settings)
    view["dispatched_at_display"] = _format_datetime_local(view.get("dispatched_at"), settings)
    for delivery_log in view.get("delivery_logs", []):
        delivery_log["sent_at_display"] = _format_datetime_local(delivery_log.get("sent_at"), settings)
    return view


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):,.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2f}%"


def _safe_text(value: str | None, fallback: str = "evidence insufficient") -> str:
    if value is None:
        return fallback
    text = " ".join(value.split())
    return text or fallback


def _candidate_reason(session, candidate: ScreeningCandidate) -> str:
    ai_service = AIAnalysisService()
    analyses = ai_service.latest_for_target(session, target_type="screening_candidate", target_id=candidate.id)
    for analysis in analyses:
        if analysis.analysis_kind == "candidate_selection_reason" and analysis.summary:
            return _safe_text(analysis.summary)

    pattern_label = ((candidate.evidence or {}).get("pattern") or {}).get("label")
    if pattern_label:
        return _safe_text(str(pattern_label))
    if candidate.value_summary:
        return _safe_text(candidate.value_summary)
    return "evidence insufficient"


def _holding_latest_analysis(holding_read) -> str:
    if holding_read.latest_ai_analysis and holding_read.latest_ai_analysis.summary:
        return _safe_text(holding_read.latest_ai_analysis.summary)
    reasons = getattr(holding_read, "reasons", None) or []
    if reasons:
        return ", ".join(reasons)
    return "evidence insufficient"


def _build_today_rows(session, candidates: list[ScreeningCandidate], settings) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        latest_close = (candidate.evidence or {}).get("latest_close")
        rows.append(
            {
                "rank": candidate.rank_position,
                "symbol": candidate.symbol,
                "symbol_name": candidate.symbol_name or "-",
                "short_score": _format_number(candidate.overall_score),
                "treasure_score": _format_number(candidate.value_score),
                "latest_close": _format_number(latest_close),
                "entry_zone": (
                    f"{_format_number(candidate.entry_zone_low)} - {_format_number(candidate.entry_zone_high)}"
                    if candidate.entry_zone_low is not None and candidate.entry_zone_high is not None
                    else "unavailable"
                ),
                "stop_loss": _format_number(candidate.stop_loss),
                "take_profit": (
                    f"{_format_number(candidate.take_profit_1)} / {_format_number(candidate.take_profit_2)}"
                    if candidate.take_profit_1 is not None or candidate.take_profit_2 is not None
                    else "unavailable"
                ),
                "risk_reward": _format_number(candidate.risk_reward_ratio),
                "reason": _candidate_reason(session, candidate),
                "updated_at_display": _format_datetime_local(
                    (candidate.evidence or {}).get("fetched_at") or candidate.updated_at,
                    settings,
                ),
            }
        )
    return rows


def _build_treasure_rows(candidates: list[ScreeningCandidate]) -> list[dict]:
    rows: list[dict] = []
    treasure_candidates = sorted(
        [item for item in candidates if item.treasure_status == "ready"],
        key=lambda item: (item.value_score or -1.0, item.overall_score or -1.0),
        reverse=True,
    )
    for candidate in treasure_candidates:
        rows.append(
            {
                "symbol": candidate.symbol,
                "symbol_name": candidate.symbol_name or "-",
                "value_score": _format_number(candidate.value_score),
                "growth_score": _format_number(candidate.growth_score),
                "quality_score": _format_number(candidate.quality_score),
                "valuation_score": _format_number(candidate.valuation_score),
                "catalyst_score": _format_number(candidate.catalyst_score),
                "summary": _safe_text(candidate.value_summary),
                "risks": ", ".join((candidate.value_risks or {}).get("reasons", [])) or "none",
                "updated_at": candidate.updated_at.isoformat(),
            }
        )
    return rows


def _build_recent_news_rows(news_items: list[NewsItem], settings) -> list[dict]:
    rows: list[dict] = []
    for item in news_items:
        rows.append(
            {
                "symbol": item.symbol or "-",
                "title": _safe_text(item.title, fallback="-"),
                "source_name": item.source_name,
                "published_at_display": _format_datetime_local(item.published_at, settings),
            }
        )
    return rows


def _latest_delivery_attempt(report_view: dict | None) -> dict | None:
    if not report_view:
        return None
    delivery_logs = list(report_view.get("delivery_logs") or [])
    if not delivery_logs:
        return None
    latest = delivery_logs[-1]
    return {
        "attempt_no": latest.get("attempt_no"),
        "status": latest.get("status"),
        "http_status": latest.get("http_status"),
        "error_detail": latest.get("error_detail"),
        "response_body": latest.get("response_body"),
        "webhook_url_masked": latest.get("webhook_url_masked"),
        "sent_at": latest.get("sent_at"),
        "sent_at_display": latest.get("sent_at_display"),
    }


def _run_picks_job() -> None:
    with SessionLocal() as session:
        settings = build_effective_settings(session)
        rate_limits = RateLimitService(settings)
        try:
            refresh_market_data(session, trigger_source="ui_manual_screening:data_refresh")
            run = run_screening(session)
            rate_limits.record(
                session,
                operation="manual_screening_run",
                status=run.status,
                metadata={"source": "ui", "run_id": run.id},
            )
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("ui_manual_screening_failed error=%s", exc)


def _run_discord_job() -> None:
    with SessionLocal() as session:
        try:
            dispatch_prepared_daily_report(session, trigger_source="ui_manual_dispatch")
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("ui_manual_discord_dispatch_failed error=%s", exc)


def _redirect(path: str, **params: str) -> RedirectResponse:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    url = f"{path}?{query}" if query else path
    return RedirectResponse(url=url, status_code=303)


def _system_counts(session) -> dict:
    return {
        "price_bars": session.query(PriceBar).count(),
        "news_items": session.query(NewsItem).count(),
        "revenue_snapshots": session.query(RevenueSnapshot).count(),
        "fundamental_snapshots": session.query(FundamentalSnapshot).count(),
        "holdings": session.query(Holding).count(),
        "screening_runs": session.query(ScreeningRun).count(),
        "daily_reports": session.query(DailyReportRun).count(),
    }


@router.get("/", response_class=HTMLResponse)
def index() -> RedirectResponse:
    return RedirectResponse(url="/picks", status_code=302)


@router.get("/picks", response_class=HTMLResponse)
def picks_page(request: Request) -> HTMLResponse:
    with SessionLocal() as session:
        effective_settings = build_effective_settings(session)
        latest_run, candidates = get_latest_run(session)
        ready_candidates = [item for item in candidates if item.status == "ready"]
        today_rows = _build_today_rows(session, ready_candidates, effective_settings)
        latest_report = get_latest_daily_report(session)
        report_view = _build_report_view(session, latest_report, effective_settings)
        cost_snapshot = CostControlService().build_snapshot(session)
        recent_news = session.scalars(
            select(NewsItem).order_by(NewsItem.published_at.desc(), NewsItem.id.desc()).limit(10)
        ).all()

    return templates.TemplateResponse(
        request=request,
        name="today_picks.html",
        context={
            "request": request,
            "active_page": "picks",
            "page_title": "今日選股",
            "run_view": _build_run_view(latest_run, effective_settings),
            "today_rows": today_rows,
            "latest_report": report_view,
            "latest_delivery_attempt": _latest_delivery_attempt(report_view),
            "cost_snapshot": cost_snapshot,
            "recent_news": _build_recent_news_rows(recent_news, effective_settings),
            "premium_news_window_active": premium_news_window_active(effective_settings),
            "display_timezone": effective_settings.scheduler_timezone,
            "flash": request.query_params.get("flash"),
        },
    )


@router.get("/screenings", response_class=HTMLResponse)
def screenings_page(request: Request) -> HTMLResponse:
    return picks_page(request)


@router.post("/picks/run")
def run_picks(background_tasks: BackgroundTasks) -> RedirectResponse:
    with SessionLocal() as session:
        settings = build_effective_settings(session)
        rate_limits = RateLimitService(settings)
        try:
            rate_limits.enforce(
                session,
                operation="manual_screening_run",
                limit=settings.rate_limit_screening_runs_per_window,
            )
        except RateLimitExceededError as exc:
            return _redirect("/picks", flash=f"重新執行選股受到限制: {exc}")
    background_tasks.add_task(_run_picks_job)
    return _redirect("/picks", flash="Screening started in background. Refresh this page in a few seconds.")
    """
        refresh_market_data(session, trigger_source="ui_manual_screening:data_refresh")
        run_screening(session)
        rate_limits.record(session, operation="manual_screening_run", status="completed", metadata={"source": "ui"})
        session.commit()
    return _redirect("/picks", flash="已重新執行短線選股。")


    """
@router.post("/discord/run")
def run_discord_report(background_tasks: BackgroundTasks) -> RedirectResponse:
    background_tasks.add_task(_run_discord_job)
    return _redirect("/picks", flash="Discord dispatch started in background. Refresh this page for the latest result.")
    """
    with SessionLocal() as session:
        report = run_daily_screening_and_push(session, trigger_source="ui_manual")
    message = "已執行 Discord 推播。"
    if report.status == "failed":
        message = f"Discord 推播失敗: {report.error_detail or 'unknown_error'}"
    return _redirect("/picks", flash=message)


    """
@router.get("/treasures", response_class=HTMLResponse)
def treasures_page(request: Request) -> HTMLResponse:
    with SessionLocal() as session:
        latest_run, candidates = get_latest_run(session)
        treasure_rows = _build_treasure_rows(candidates)
    return templates.TemplateResponse(
        request=request,
        name="treasures.html",
        context={
            "request": request,
            "active_page": "treasures",
            "page_title": "寶藏股",
            "run": latest_run,
            "treasure_rows": treasure_rows,
            "flash": request.query_params.get("flash"),
        },
    )


@router.get("/holdings", response_class=HTMLResponse)
def holdings_page(request: Request) -> HTMLResponse:
    with SessionLocal() as session:
        holdings = session.scalars(select(Holding).order_by(Holding.created_at.desc())).all()
        holding_rows = []
        for holding in holdings:
            enriched = enrich_holding(session, holding)
            latest_close = enriched.latest_close
            return_pct = None
            if latest_close is not None and holding.average_cost:
                return_pct = ((latest_close - holding.average_cost) / holding.average_cost) * 100
            holding_rows.append(
                {
                    "id": holding.id,
                    "symbol": holding.symbol,
                    "symbol_name": enriched.symbol_name or "-",
                    "average_cost": _format_number(holding.average_cost),
                    "latest_close": _format_number(latest_close),
                    "return_pct": _format_pct(return_pct),
                    "trend_status": enriched.trend_status,
                    "alert_status": enriched.alert_status,
                    "action": enriched.action,
                    "latest_analysis": _holding_latest_analysis(enriched),
                }
            )
        session.commit()
    return templates.TemplateResponse(
        request=request,
        name="holdings.html",
        context={
            "request": request,
            "active_page": "holdings",
            "page_title": "我的持股",
            "holding_rows": holding_rows,
            "flash": request.query_params.get("flash"),
        },
    )


@router.post("/holdings/add")
async def add_holding(request: Request) -> RedirectResponse:
    form = await request.form()
    with SessionLocal() as session:
        holding = Holding(
            symbol=str(form.get("symbol", "")).strip(),
            symbol_name=str(form.get("symbol_name", "")).strip() or None,
            quantity=int(form.get("quantity", 0)),
            average_cost=float(form.get("average_cost", 0)),
            opened_date=date.fromisoformat(str(form.get("opened_date"))) if form.get("opened_date") else date.today(),
            custom_stop_loss=float(form.get("custom_stop_loss")) if form.get("custom_stop_loss") else None,
            custom_target_price=float(form.get("custom_target_price")) if form.get("custom_target_price") else None,
            note=str(form.get("note", "")).strip() or None,
        )
        session.add(holding)
        session.commit()
    return _redirect("/holdings", flash="已新增持股。")


@router.post("/holdings/refresh")
def refresh_holdings() -> RedirectResponse:
    with SessionLocal() as session:
        symbols = session.scalars(select(Holding.symbol)).all()
        refresh_symbols_data(
            session,
            symbols=list(symbols),
            trigger_source="ui_holdings_refresh:data_refresh",
            force_refresh=True,
        )
        PositionMonitorService().monitor_all_positions(session)
    return _redirect("/holdings", flash="已重新更新持股監控。")


@router.post("/system/refresh-data")
def refresh_system_data() -> RedirectResponse:
    with SessionLocal() as session:
        refresh_market_data(session, trigger_source="ui_system_refresh:data_refresh", force_refresh=True)
    return _redirect("/system", flash="Market data refresh completed.")


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    with SessionLocal() as session:
        ui_settings = get_settings_for_ui(session)
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={
            "request": request,
            "active_page": "settings",
            "page_title": "設定",
            "settings_sections": ui_settings["sections"],
            "weight_sum": ui_settings["weight_sum"],
            "schedule_preview": ui_settings["schedule_preview"],
            "flash": request.query_params.get("flash"),
        },
    )


@router.post("/settings/save")
async def save_settings_page(request: Request) -> RedirectResponse:
    form = await request.form()
    payload: dict[str, str] = {}
    checkbox_keys = {"ai_enabled", "news_analysis_enabled"}
    for key in checkbox_keys:
        payload[key] = "true" if form.get(key) else "false"
    for key, value in form.items():
        if key in checkbox_keys:
            continue
        payload[key] = str(value)

    with SessionLocal() as session:
        try:
            changed = save_settings(session, payload)
        except ValueError as exc:
            return _redirect("/settings", flash=f"設定儲存失敗: {exc}")

    suffix = "設定已儲存。"
    if {"screening_hour", "screening_minute"} & set(changed):
        suffix = "設定已儲存。排程時間會在 worker 重新啟動後套用到下一次 cron。"
    return _redirect("/settings", flash=suffix)


@router.get("/system", response_class=HTMLResponse)
def system_status_page(request: Request) -> HTMLResponse:
    with SessionLocal() as session:
        effective_settings = build_effective_settings(session)
        latest_run = session.scalar(select(ScreeningRun).order_by(desc(ScreeningRun.created_at)))
        latest_refresh = session.scalar(select(DataRefreshRun).order_by(desc(DataRefreshRun.created_at)))
        latest_report = get_latest_daily_report(session)
        report_view = _build_report_view(session, latest_report, effective_settings)
        counts = _system_counts(session)
        ui_settings = get_settings_for_ui(session)
        cost_snapshot = CostControlService().build_snapshot(session)
        flag_service = FeatureFlagService()
        feature_flags = flag_service.describe(session)
        cost_dashboard_enabled = flag_service.is_enabled("cost_dashboard", session)
        startup_check = StartupCheckService().build_snapshot(session)

    return templates.TemplateResponse(
        request=request,
        name="system_status.html",
        context={
            "request": request,
            "active_page": "system",
            "page_title": "系統狀態",
            "counts": counts,
            "latest_run": _build_run_view(latest_run, effective_settings),
            "latest_refresh": _build_refresh_view(latest_refresh, effective_settings),
            "latest_report": report_view,
            "latest_delivery_attempt": _latest_delivery_attempt(report_view),
            "schedule_preview": ui_settings["schedule_preview"],
            "cost_snapshot": cost_snapshot,
            "feature_flags": feature_flags,
            "cost_dashboard_enabled": cost_dashboard_enabled,
            "startup_check": startup_check,
            "display_timezone": effective_settings.scheduler_timezone,
            "flash": request.query_params.get("flash"),
        },
    )

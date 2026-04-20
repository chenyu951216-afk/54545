from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from tw_stock_ai.models import Holding, NewsItem, PositionAlert, PriceBar, SecurityProfile
from tw_stock_ai.schemas import HoldingRead, PositionAlertRead
from tw_stock_ai.services.alert_rules import build_position_alerts
from tw_stock_ai.services.exit_analysis import build_exit_analysis
from tw_stock_ai.services.serialization import to_jsonable
from tw_stock_ai.services.trend_health import analyze_trend_health


def _resolve_symbol_name(session: Session, holding: Holding, bars: list[PriceBar]) -> str | None:
    if holding.symbol_name:
        return holding.symbol_name
    profile = session.scalar(select(SecurityProfile).where(SecurityProfile.symbol == holding.symbol))
    if profile and profile.name:
        return profile.name
    for bar in reversed(sorted(bars, key=lambda item: item.trade_date)):
        payload = bar.raw_payload or {}
        for key in ("symbol_name", "name", "stock_name", "company_name"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class PositionMonitorService:
    def monitor_position(self, session: Session, holding: Holding) -> HoldingRead:
        from tw_stock_ai.config import get_settings

        settings = get_settings()
        bars = session.scalars(
            select(PriceBar)
            .where(PriceBar.symbol == holding.symbol)
            .order_by(PriceBar.trade_date.asc())
        ).all()
        latest_bar = bars[-1] if bars else None
        news_items = session.scalars(
            select(NewsItem)
            .where(NewsItem.symbol == holding.symbol)
            .order_by(desc(NewsItem.published_at))
            .limit(20)
        ).all()

        trend = analyze_trend_health(bars, settings)
        alerts = build_position_alerts(
            holding=holding,
            bars=bars,
            trend=trend,
            news_items=news_items,
            settings=settings,
        )
        exit_analysis = build_exit_analysis(
            trend_status=trend["trend_status"],
            alerts=alerts,
            settings=settings,
        )

        latest_close = float(latest_bar.close) if latest_bar else None
        unrealized_pnl = (
            round((latest_close - holding.average_cost) * holding.quantity, 2)
            if latest_close is not None
            else None
        )

        symbol_name = _resolve_symbol_name(session, holding, bars)
        evidence = {
            "symbol": holding.symbol,
            "latest_trade_date": latest_bar.trade_date.isoformat() if latest_bar else None,
            "latest_close": latest_close,
            "source_name": latest_bar.source_name if latest_bar else None,
            "source_url": latest_bar.source_url if latest_bar else None,
            "fetched_at": latest_bar.fetched_at.isoformat() if latest_bar else None,
            "trend_metrics": trend["metrics"],
            "alerts": alerts,
        }

        holding.symbol_name = symbol_name
        holding.current_trend = trend["trend_status"]
        holding.alert_status = "active" if alerts else "none"
        holding.last_action = exit_analysis["action"]
        holding.last_confidence = exit_analysis["confidence"]
        holding.last_monitor_reasons = to_jsonable({"reasons": exit_analysis["reasons"]})
        holding.last_monitor_evidence = to_jsonable(evidence)
        holding.last_monitored_at = datetime.now(timezone.utc)

        session.add(holding)
        self._replace_active_alerts(session, holding, alerts)
        session.flush()

        from tw_stock_ai.services.ai_analysis import AIAnalysisService

        ai_service = AIAnalysisService()
        ai_analyses = ai_service.latest_for_target(session, target_type="holding", target_id=holding.id)
        latest_ai = ai_analyses[0] if ai_analyses else None

        alert_reads = [
            PositionAlertRead.model_validate(item)
            for item in session.scalars(
                select(PositionAlert)
                .where(
                    PositionAlert.holding_id == holding.id,
                    PositionAlert.status == "active",
                )
                .order_by(PositionAlert.triggered_at.desc())
            ).all()
        ]

        return HoldingRead.model_validate(
            {
                **holding.__dict__,
                "symbol_name": symbol_name,
                "latest_close": latest_close,
                "unrealized_pnl": unrealized_pnl,
                "trend_status": trend["trend_status"],
                "exit_signal": trend["exit_signal"],
                "action": exit_analysis["action"],
                "confidence": exit_analysis["confidence"],
                "reasons": exit_analysis["reasons"],
                "alert_status": holding.alert_status or "none",
                "alerts": [item.model_dump() for item in alert_reads],
                "latest_ai_analysis": latest_ai.model_dump() if latest_ai else None,
                "evidence": evidence,
                "ai_analyses": [item.model_dump() for item in ai_analyses],
            }
        )

    def monitor_all_positions(self, session: Session) -> list[HoldingRead]:
        holdings = session.scalars(select(Holding).order_by(Holding.created_at.desc())).all()
        results = [self.monitor_position(session, holding) for holding in holdings]
        session.commit()
        return results

    def _replace_active_alerts(self, session: Session, holding: Holding, alerts: list[dict]) -> None:
        active_alerts = session.scalars(
            select(PositionAlert).where(
                PositionAlert.holding_id == holding.id,
                PositionAlert.status == "active",
            )
        ).all()
        for item in active_alerts:
            item.status = "resolved"
            item.resolved_at = datetime.now(timezone.utc)
            session.add(item)

        for alert in alerts:
            session.add(
                PositionAlert(
                    holding_id=holding.id,
                    symbol=holding.symbol,
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    status="active",
                    message=alert["message"],
                    evidence=to_jsonable(alert["evidence"]),
                    triggered_at=alert["triggered_at"],
                )
            )

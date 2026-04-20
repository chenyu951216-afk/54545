from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.models import AIAnalysisRecord, UsageEvent
from tw_stock_ai.services.feature_flags import FeatureFlagService
from tw_stock_ai.services.usage_tracking import UsageTracker


class CostControlService:
    def __init__(self, settings: Settings | None = None, tracker: UsageTracker | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracker = tracker or UsageTracker()
        self.flags = FeatureFlagService(self.settings)

    def build_snapshot(self, session: Session) -> dict:
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        ai_actual_cost = float(
            session.scalar(
                select(func.coalesce(func.sum(AIAnalysisRecord.estimated_cost_twd), 0.0)).where(
                    AIAnalysisRecord.generated_at >= month_start
                )
            )
            or 0.0
        )
        non_ai_tracked_cost = self._event_costs(session, month_start, exclude_event_type="ai_call")
        monthly_total_estimated = round(ai_actual_cost + non_ai_tracked_cost, 6)
        monthly_budget = float(self.settings.overall_monthly_budget_twd)
        ai_budget = float(self.settings.ai_monthly_budget_twd)

        return {
            "monthly_budget_twd": monthly_budget,
            "monthly_estimated_cost_twd": monthly_total_estimated,
            "monthly_remaining_budget_twd": round(monthly_budget - monthly_total_estimated, 6),
            "ai_budget_twd": ai_budget,
            "ai_actual_cost_twd": round(ai_actual_cost, 6),
            "daily_usage": {
                "api_calls": self._event_count(session, day_start, "api_call"),
                "external_api_calls": self._event_count(session, day_start, "external_api_call"),
                "ai_calls": self._event_count(session, day_start, "ai_call"),
                "notifications": self._event_count(session, day_start, "notification_send"),
            },
            "monthly_usage": {
                "api_calls": self._event_count(session, month_start, "api_call"),
                "external_api_calls": self._event_count(session, month_start, "external_api_call"),
                "ai_calls": self._event_count(session, month_start, "ai_call"),
                "notifications": self._event_count(session, month_start, "notification_send"),
            },
            "feature_flags": self.flags.describe(session),
            "guardrails_enabled": self.flags.is_enabled("cost_guardrails", session),
        }

    def within_overall_budget(self, session: Session, extra_cost: float = 0.0) -> bool:
        snapshot = self.build_snapshot(session)
        return snapshot["monthly_estimated_cost_twd"] + extra_cost <= self.settings.overall_monthly_budget_twd

    def _event_count(self, session: Session, since: datetime, event_type: str) -> int:
        total = session.scalar(
            select(func.count(UsageEvent.id)).where(
                UsageEvent.event_type == event_type,
                UsageEvent.occurred_at >= since,
            )
        )
        return int(total or 0)

    def _event_costs(self, session: Session, since: datetime, exclude_event_type: str | None = None) -> float:
        query = select(func.coalesce(func.sum(UsageEvent.estimated_cost_twd), 0.0)).where(
            UsageEvent.occurred_at >= since,
        )
        if exclude_event_type is not None:
            query = query.where(UsageEvent.event_type != exclude_event_type)
        total = session.scalar(query)
        return float(total or 0.0)

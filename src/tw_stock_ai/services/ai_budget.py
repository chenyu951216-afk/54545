from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings
from tw_stock_ai.models import AIAnalysisRecord


def current_month_cost_twd(session: Session) -> float:
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = session.scalar(
        select(func.coalesce(func.sum(AIAnalysisRecord.estimated_cost_twd), 0.0)).where(
            AIAnalysisRecord.generated_at >= month_start
        )
    )
    return float(total or 0.0)


def within_budget(session: Session, settings: Settings, extra_cost: float = 0.0) -> bool:
    return current_month_cost_twd(session) + extra_cost <= settings.ai_monthly_budget_twd

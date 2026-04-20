from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tw_stock_ai.models import UsageEvent
from tw_stock_ai.services.serialization import to_jsonable


class UsageTracker:
    def record(
        self,
        session: Session,
        *,
        event_type: str,
        operation: str,
        status: str,
        provider: str | None = None,
        units: int = 1,
        estimated_cost_twd: float = 0.0,
        metadata: dict | None = None,
        occurred_at: datetime | None = None,
    ) -> UsageEvent:
        event = UsageEvent(
            event_type=event_type,
            provider=provider,
            operation=operation,
            status=status,
            units=units,
            estimated_cost_twd=estimated_cost_twd,
            metadata_json=to_jsonable(metadata or {}),
            occurred_at=occurred_at or datetime.now(timezone.utc),
        )
        session.add(event)
        session.flush()
        return event

    def count_since(
        self,
        session: Session,
        *,
        operation: str,
        since: datetime,
        event_type: str | None = None,
        status: str | None = None,
    ) -> int:
        query = select(func.count(UsageEvent.id)).where(
            UsageEvent.operation == operation,
            UsageEvent.occurred_at >= since,
        )
        if event_type is not None:
            query = query.where(UsageEvent.event_type == event_type)
        if status is not None:
            query = query.where(UsageEvent.status == status)
        total = session.scalar(query)
        return int(total or 0)

    def sum_cost_since(
        self,
        session: Session,
        *,
        since: datetime,
        event_type: str | None = None,
    ) -> float:
        query = select(func.coalesce(func.sum(UsageEvent.estimated_cost_twd), 0.0)).where(
            UsageEvent.occurred_at >= since,
        )
        if event_type is not None:
            query = query.where(UsageEvent.event_type == event_type)
        total = session.scalar(query)
        return float(total or 0.0)

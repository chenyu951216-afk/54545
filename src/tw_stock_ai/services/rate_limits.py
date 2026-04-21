from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.services.usage_tracking import UsageTracker


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    current_count: int
    limit: int
    window_minutes: int
    operation: str


class RateLimitExceededError(RuntimeError):
    pass


class RateLimitService:
    def __init__(self, settings: Settings | None = None, tracker: UsageTracker | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracker = tracker or UsageTracker()

    def evaluate(self, session: Session, *, operation: str, limit: int) -> RateLimitDecision:
        since = datetime.now(timezone.utc) - timedelta(minutes=self.settings.api_rate_limit_window_minutes)
        current_count = self.tracker.count_since(session, operation=operation, since=since, event_type="api_call")
        return RateLimitDecision(
            allowed=current_count < limit,
            current_count=current_count,
            limit=limit,
            window_minutes=self.settings.api_rate_limit_window_minutes,
            operation=operation,
        )

    def enforce(self, session: Session, *, operation: str, limit: int) -> RateLimitDecision:
        decision = self.evaluate(session, operation=operation, limit=limit)
        if not decision.allowed:
            raise RateLimitExceededError(
                f"rate limit exceeded for {operation}: {decision.current_count}/{decision.limit} in "
                f"{decision.window_minutes} minutes"
            )
        return decision

    def record(self, session: Session, *, operation: str, status: str = "completed", metadata: dict | None = None) -> None:
        self.tracker.record(
            session,
            event_type="api_call",
            operation=operation,
            provider="internal",
            status=status,
            metadata=metadata,
        )

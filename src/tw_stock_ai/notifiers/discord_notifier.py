from __future__ import annotations

from sqlalchemy.orm import Session

from tw_stock_ai.config import Settings, get_settings
from tw_stock_ai.models import DailyReportRun
from tw_stock_ai.notifiers.base import BaseNotifier, NotificationResult
from tw_stock_ai.services.discord import DiscordWebhookSender


class DiscordNotifier(BaseNotifier):
    channel_name = "discord"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sender = DiscordWebhookSender(settings=self.settings)

    def send(self, session: Session, *, report_run: DailyReportRun) -> NotificationResult:
        result = self.sender.send_report(session, report_run)
        return NotificationResult(status=result.status, detail=result.detail, attempts=result.attempts)

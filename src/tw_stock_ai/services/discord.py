from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib import error, request

from sqlalchemy.orm import Session

from tw_stock_ai.config import get_settings
from tw_stock_ai.models import DailyReportRun, DiscordDeliveryLog
from tw_stock_ai.services.logging_config import get_logger
from tw_stock_ai.services.usage_tracking import UsageTracker

logger = get_logger("tw_stock_ai.discord")


@dataclass(slots=True)
class DiscordSendResult:
    status: str
    attempts: int
    http_status: int | None = None
    detail: str | None = None


class DiscordWebhookSender:
    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.usage_tracker = UsageTracker()

    def send_report(self, session: Session, report_run: DailyReportRun) -> DiscordSendResult:
        masked_url = self._mask_webhook_url(self.settings.discord_webhook_url)
        payload = {"content": report_run.rendered_content}

        if not self.settings.discord_enabled:
            self._log_attempt(
                session,
                report_run=report_run,
                attempt_no=1,
                status="skipped",
                webhook_url_masked=masked_url,
                payload_json=payload,
                error_detail="discord_disabled",
            )
            report_run.status = "skipped"
            report_run.error_detail = "discord_disabled"
            self._record_usage(session, status="skipped", report_run=report_run)
            return DiscordSendResult(status="skipped", attempts=1, detail="discord_disabled")

        if not self.settings.discord_webhook_url:
            self._log_attempt(
                session,
                report_run=report_run,
                attempt_no=1,
                status="skipped",
                webhook_url_masked=masked_url,
                payload_json=payload,
                error_detail="DISCORD_WEBHOOK_URL is not configured",
            )
            report_run.status = "skipped"
            report_run.error_detail = "DISCORD_WEBHOOK_URL is not configured"
            self._record_usage(session, status="skipped", report_run=report_run)
            return DiscordSendResult(status="skipped", attempts=1, detail=report_run.error_detail)

        last_error: str | None = None
        last_http_status: int | None = None
        attempts_made = 0
        for attempt_no in range(1, self.settings.discord_retry_attempts + 1):
            attempts_made = attempt_no
            try:
                response_status, response_body = self._post_payload(payload)
                self._log_attempt(
                    session,
                    report_run=report_run,
                    attempt_no=attempt_no,
                    status="sent",
                    webhook_url_masked=masked_url,
                    http_status=response_status,
                    response_body=response_body,
                    payload_json=payload,
                )
                report_run.status = "sent"
                report_run.error_detail = None
                report_run.dispatched_at = datetime.now(timezone.utc)
                self._record_usage(session, status="sent", report_run=report_run)
                logger.info("discord_report_sent report_run_id=%s attempt=%s", report_run.id, attempt_no)
                return DiscordSendResult(
                    status="sent",
                    attempts=attempt_no,
                    http_status=response_status,
                )
            except error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                detail_suffix = f":{response_body[:200].strip()}" if response_body else ""
                last_error = f"http_error:{exc.code}{detail_suffix}"
                last_http_status = exc.code
                self._log_attempt(
                    session,
                    report_run=report_run,
                    attempt_no=attempt_no,
                    status="failed",
                    webhook_url_masked=masked_url,
                    http_status=exc.code,
                    response_body=response_body,
                    payload_json=payload,
                    error_detail=last_error,
                )
                if not self._should_retry_http_status(exc.code):
                    break
            except error.URLError as exc:
                last_error = f"url_error:{exc.reason}"
                self._log_attempt(
                    session,
                    report_run=report_run,
                    attempt_no=attempt_no,
                    status="failed",
                    webhook_url_masked=masked_url,
                    payload_json=payload,
                    error_detail=last_error,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"unexpected_error:{exc}"
                self._log_attempt(
                    session,
                    report_run=report_run,
                    attempt_no=attempt_no,
                    status="failed",
                    webhook_url_masked=masked_url,
                    payload_json=payload,
                    error_detail=last_error,
                )

            if attempt_no < self.settings.discord_retry_attempts:
                time.sleep(self.settings.discord_retry_backoff_seconds)

        report_run.status = "failed"
        report_run.error_detail = last_error
        self._record_usage(session, status="failed", report_run=report_run)
        logger.warning("discord_report_failed report_run_id=%s detail=%s", report_run.id, last_error)
        return DiscordSendResult(
            status="failed",
            attempts=attempts_made,
            http_status=last_http_status,
            detail=last_error,
        )

    def _post_payload(self, payload: dict) -> tuple[int, str]:
        encoded_payload = json.dumps(payload).encode("utf-8")
        webhook_url = self._normalize_webhook_url(self.settings.discord_webhook_url)
        req = request.Request(
            webhook_url,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/plain, */*",
                "User-Agent": "tw-stock-ai/1.0 (+discord-webhook-client)",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.settings.discord_timeout_seconds) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body

    def _log_attempt(
        self,
        session: Session,
        *,
        report_run: DailyReportRun,
        attempt_no: int,
        status: str,
        webhook_url_masked: str | None,
        payload_json: dict,
        http_status: int | None = None,
        response_body: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        session.add(
            DiscordDeliveryLog(
                report_run_id=report_run.id,
                attempt_no=attempt_no,
                status=status,
                webhook_url_masked=webhook_url_masked,
                http_status=http_status,
                response_body=response_body,
                error_detail=error_detail,
                payload_json=payload_json,
                sent_at=datetime.now(timezone.utc),
            )
        )
        session.flush()

    def _mask_webhook_url(self, webhook_url: str | None) -> str | None:
        if not webhook_url:
            return None
        if len(webhook_url) <= 16:
            return webhook_url
        return f"{webhook_url[:8]}...{webhook_url[-8:]}"

    def _normalize_webhook_url(self, webhook_url: str | None) -> str:
        if not webhook_url:
            raise ValueError("discord_webhook_url_missing")
        return webhook_url.replace("https://discordapp.com/", "https://discord.com/")

    def _record_usage(self, session: Session, *, status: str, report_run: DailyReportRun) -> None:
        self.usage_tracker.record(
            session,
            event_type="notification_send",
            operation="discord_report_send",
            provider="discord",
            status=status,
            estimated_cost_twd=self.settings.estimated_notification_cost_per_send_twd,
            metadata={"report_run_id": report_run.id, "report_date": report_run.report_date.isoformat()},
        )

    def _should_retry_http_status(self, http_status: int) -> bool:
        if http_status == 429 or http_status >= 500:
            return True
        return False
